"""Run every available trigger-trained confirmation window through the P4 QEMU five-class kernel."""
import argparse
import ctypes
from hashlib import sha256
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import train_positive_readout as fit
import proposal_readout_onsets as gating
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import boot
from run_c161_coded_cue_p4_v5 import QEMU
from score_proposal_readout import GROUPS, base

BASE = RESULTS/(fit.STEM+"-p4")


def collect(model):
    lib = base.BUILD/"frontend.so"
    assert sha(lib) == read(base.BUILD/"build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    lookup = base.poll.Model(model).table
    blocks, rows, streams, files = [], [], [], {}
    for group in GROUPS:
        traces = RESULTS/f"{fit.STEM}-{group}.json"
        ons = RESULTS/f"{gating.STEM}-{group}.json"
        files[group] = dict(host=sha(traces), onset=sha(ons))
        for (_,pcm,c161,_,_), r, o in zip(base.sources(group), read(traces)["records"],read(ons)["records"],strict=True):
            assert base.poll.digest(pcm) == r["pcm_sha256"] == o["pcm_sha256"]
            stream = dict(group=group, record=r["index"], mic_sha256=r["pcm_sha256"], c161=c161,
                          host_events=r["routes"]["trigger"], host_onset_events=o["routes"]["trigger_onset"], windows=[])
            if group in ("acoustics","matched"):
                stream.update(source=r["source"])
            streams.append(stream)
            if not r["checks"]:
                continue
            raw,_,clock = base.native_inputs(native,pcm)
            for c,d in zip(r["checks"],o["checks"],strict=True):
                t = c["end"]
                assert t == d["end"] and c["proposal"] == d["proposal"]
                x = lookup[np.rint(fit.parent.room.window(raw,t).clip(0,666)*(255./666)).astype(np.uint8)]
                y = np.asarray(c["codes"]["trigger"],dtype=np.int8)
                stream["windows"].append(len(rows))
                rows.append(dict(end=t, host_codes=y.tolist(), input_sha256=base.poll.digest(x), in_window=d["in_window"]))
                blocks.append(x.tobytes()+y.tobytes())
            del raw
    return blocks, rows, streams, files


def main(build):
    exp = read(RESULTS/(fit.STEM+"-exports.json"))["models"]["trigger"]
    model = RESULTS/exp["file"]
    assert sha(model)==exp["sha256"]
    images = read(build/"flasher_args.json")["flash_files"]
    assert sum((build/f).read_bytes().count(model.read_bytes()) for f in images.values()) == 1
    cfg = read(build/"config/sdkconfig.json")
    assert cfg["NN_ANSI_C"] and cfg["IDF_TARGET"]=="esp32p4"
    blocks,rows,streams,files = collect(model)
    archive = BASE.with_suffix(".zip")
    batches = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        work = Path(tmp)
        for start in range(0,len(blocks),150):
            batch = blocks[start:start+150]
            data = len(batch).to_bytes(4,"little")+b"".join(batch)
            uart = work/f"{start:04}-uart.log"
            boot(build,uart,work/"flash.bin",b"ONSET_DONE",(2<<20,data))
            text = uart.read_text()
            assert "ONSET_FC per_channel" in text and "ONSET_FAIL" not in text
            values = [list(map(int,line.split())) for line in re.findall(r"^ONSET_ROW ([0-9 -]+)$",text,re.M)]
            count,heap = map(int,re.search(r"ONSET_END count=(\d+) min_internal=(\d+)",text).groups())
            arena,init = map(int,re.search(r"ONSET_READY arena=(\d+) init_us=(\d+)",text).groups())
            assert count == len(batch) == len(values)
            for i,v in enumerate(values):
                assert len(v)==7 and v[0]==i and v[-1]>0
                rows[start+i].update(device_codes=v[1:6],invoke_us=v[-1])
            batches.append(dict(start=start,count=count,arena_used=arena,min_internal_free=heap,init_us=init,
                                fixture_sha256=sha256(data).hexdigest(),uart_sha256=sha(uart)))
            z.write(uart,uart.name)
    archive.with_suffix(".tmp").replace(archive)
    for s in streams:
        for gated,key in ((False,"events"),(True,"onset_events")):
            complete = 0
            notices = []
            for idx in s["windows"]:
                win = rows[idx]
                if gated and not win["in_window"]:
                    continue
                complete = max(win["end"],complete)+(win["invoke_us"]*16+999)//1000
                if win["device_codes"][0]>=0:
                    notices.append(complete)
            s[key] = base.scoring.merged(s["c161"],notices)
    save(BASE.with_suffix(".json"),dict(records=rows,streams=streams,batches=batches,source_files=files,
        mismatches=sum(r["host_codes"]!=r["device_codes"] for r in rows),sha256=sha(model),
        qemu_sha256=sha(QEMU),archive_sha256=sha(archive),images={f:sha(build/f) for f in images.values()},
        script_sha256=sha(Path(__file__)),policy="All complete proposals, preserved across fresh, daily, Bot, "
        "continuous conversation, four acoustics, paired Willow scenes, 12 music tracks and 66 background streams. "
        "Primed frontend and scheduler on host; P4 QEMU per-channel FC evaluates the exact INT8 features. Charge "
        "measured serial Invoke time only at window availability; C161 events keep their recorded clocks."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("build",type=Path)
    main(p.parse_args().build)
