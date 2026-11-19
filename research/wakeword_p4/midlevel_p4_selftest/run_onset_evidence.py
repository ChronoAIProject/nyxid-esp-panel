"""Compare real P4-QEMU inference with host INT8 codes and replay the measured worker delay."""
import argparse
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
import check_c161_onset_int8 as host
from check_c161_pocketsphinx_first_command import load_reader
from run_c161_first_command_v5 import boot
from run_c161_coded_cue_p4_v5 import QEMU
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav


def main(build):
    stem = host.exported.STEM + "-p4"
    report = host.read(host.REPORT)
    fixture = host.FIXTURE
    assert host.sha(fixture) == report["fixture"]["sha256"]
    data = fixture.read_bytes()
    assert len(data) == 4+5765*int.from_bytes(data[:4],"little")
    assert host.sha(host.exported.MODEL) == report["model"]
    images = host.read(build/"flasher_args.json")["flash_files"]
    model_bytes = host.exported.MODEL.read_bytes()
    assert sum((build/n).read_bytes().count(model_bytes) for n in images.values()) == 1
    config = host.read(build/"config/sdkconfig.json")
    assert config["IDF_TARGET"] == "esp32p4" and config["NN_ANSI_C"]
    uart = host.evidence.WORK/"p4-uart.log"
    boot(build=build, log=uart, image=host.evidence.WORK/"p4-flash.bin", marker=b"ONSET_DONE", flash_payload=(2<<20,data))
    text = uart.read_text(errors="replace")
    match = re.search(r"ONSET_END count=(\d+) min_internal=(\d+)",text)
    assert match, text[-2000:]
    count, heap = map(int,match.groups())
    arena, init = map(int,re.search(r"ONSET_READY arena=(\d+) init_us=(\d+)",text).groups())
    rows = [list(map(int,m.split())) for m in re.findall(r"ONSET_ROW ([0-9 -]+)",text)]
    assert count == report["fixture"]["windows"] == len(rows)
    raw = [r for r in report["records"] if r["cohort"] == "appointment"]
    parent = host.read(host.RESULTS/f"{host.evidence.STEM}-evaluation.json")
    ref = {r["pcm_sha256"]:r for r in parent["records"] if r["cohort"] == "appointment"}
    reader, wh = load_reader()
    assert wh == report["asr"]
    i = 0
    records = []
    for r,src in zip(raw,host.read(host.evidence.PLAN)["records"],strict=True):
        pcm = load_wav(src)
        assert host.digest(pcm) == r["pcm_sha256"]
        end = 0
        accept, worker = [], []
        for window in r["checks"]:
            index, *codes, micros = rows[i]
            assert index == i and micros > 0 and len(codes)==5
            start = 4+5765*i
            assert host.digest(np.frombuffer(data[start:start+5760], dtype="int8")) == window["input_sha256"]
            expected = [v if v < 128 else v-256 for v in data[start+5760:start+5765]]
            assert expected == window["codes"]
            i += 1
            ready = window["available"]
            begin = max(end,ready)
            end = begin+(micros*16+999)//1000
            worker.append(dict(index=index,available=ready,begin=begin,end=end,invoke_us=micros,
                               device_codes=codes,host_codes=expected))
            if codes[0]>=0:
                accept.append(end)
        events = host.grammar.merged(ref[r["pcm_sha256"]]["events"]["c161"],accept)
        ann = annotate(src,pcm,dict(device_delay=events),[],[],reader)
        for row in ann["rows"]:
            del row["native_proposals"],row["expanded_proposals"]
        records.append(dict(key=r["key"],pcm_sha256=r["pcm_sha256"],worker=worker,events=events,**ann))
    assert i==count
    log = host.RESULTS/(stem+"-uart.log")
    log.write_bytes(uart.read_bytes())
    host.save(host.RESULTS/(stem+".json"),dict(records=records,fixture=report["fixture"],arena_used=arena,
         arena_reserved=192*1024,init_us=init,min_internal_free=heap, model_sha256=report["model"],
         asr=wh,host_replay_sha256=host.sha(host.REPORT),uart_sha256=host.sha(log),qemu_sha256=host.sha(QEMU),
         code={p.name:host.sha(p) for p in (Path(__file__),Path(__file__).parent/"main/onset_evidence.cpp")},
         images={n:dict(sha256=host.sha(build/n),bytes=(build/n).stat().st_size) for n in images.values()},
         policy="ESP32-P4 QEMU esp-nn ANSI C, one serialized inference worker. VAD and features supplied from the host. Only Invoke charged at observed window availability. Original C161 timing reused. Endpoints and ASR as in host evaluation."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build",type=Path)
    main(parser.parse_args().build)
