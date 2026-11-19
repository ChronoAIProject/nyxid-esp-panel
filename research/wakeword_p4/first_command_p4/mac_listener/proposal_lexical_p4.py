"""Run all context/noise proposals and the accepted background window in the existing Q8 P4 probe."""
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import proposal_lexical_readout as host
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_p4_first_command import sample_to_us, service, us_to_sample
from replay_c161_q8_keyword_map_p4 import run_qemu

STEM = host.STEM+"-p4"
SIM = Path(".local/wakeword/c161-q8-name-contrast/p4")
BASELINE = RESULTS/"4090-c161-q8-name-contrast-p4.json"
IDF = Path(".local/wakeword/esp-idf/components/fatfs/fatfsgen.py")
PYTHON = Path(".local/wakeword/idf-tools/python_env/idf5.5_py3.11_env/bin/python")


def main():
    inputs = read(SIM/"q8/inputs.json")
    old = read(BASELINE)
    assert sha(SIM/"q8/inputs.json") == old["input_sha256"]
    assert sha(run_qemu.QEMU) == old["qemu_sha256"]
    rows, streams, sources = [], {}, {}
    with TemporaryDirectory() as td:
        work = Path(td)
        cases = work/"cases"
        cases.mkdir()
        q8 = work/"q8"
        q8.mkdir()
        build = work/"build-q8"
        build.mkdir()
        with (cases/"cases.bin").open("xb") as f:
            for group in ("context","acoustics","background"):
                path = RESULTS/f"{host.STEM}-{group}.json"
                sources[group] = sha(path)
                for (meta,pcm,c161,_,_), r in zip(host.base.sources(group),read(path)["records"],strict=True):
                    assert host.base.poll.digest(pcm) == r["pcm_sha256"] and c161 == r["c161"]
                    if group == "acoustics" and meta["acoustic"] != "noise":
                        continue
                    key = str(r["index"])
                    info = dict(index=r["index"], pcm_sha256=r["pcm_sha256"], c161=c161, host=r["routes"]["name-contrast"],
                                source=r["source"], windows=[])
                    if group == "background":
                        info["key"] = r["key"]
                        if not info["host"]:
                            continue
                    streams[f"{group}/{key}"] = info
                    for c in r["checks"]["name-contrast"]:
                        if group == "background" and c["confirmed"] is None:
                            continue
                        chunk = pcm[c["start"]:c["end"]].astype("<i2",copy=False)
                        f.write(struct.pack("<III",c["start"],c["end"],c["proposal"]))
                        f.write(chunk.tobytes())
                        info["windows"].append(len(rows))
                        rows.append(dict(cohort=group,key=key,window_pcm_sha256=host.base.poll.digest(chunk),**c))
        assert len(rows) == 62
        inputs = dict(inputs,records=rows,files=dict(inputs["files"],**{"cases.bin":sha(cases/"cases.bin")}))
        (q8/"inputs.json").write_text(json.dumps(inputs,indent=2)+"\n")
        (q8/"volume").symlink_to((SIM/"q8/volume").resolve())
        (work/"q8-sdkconfig").symlink_to((SIM/"q8-sdkconfig").resolve())
        original = SIM/"build-q8"
        for f in [*old["images"],"nyx_lexical_probe.elf","flasher_args.json"]:
            if f == "cases.bin":
                continue
            src = original/f
            if f in old["images"]:
                assert sha(src) == old["images"][f]
            dest = build/f
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(src.resolve())
        assert sha(build/"nyx_lexical_probe.elf") == old["elf_sha256"]
        subprocess.run([str(PYTHON),str(IDF),str(cases),"--long_name_support","--use_default_datetime",
                        "--partition_size","0xbf0000","--output_file",str(build/"cases.bin"),"--sector_size","4096"],check=True)
        run_qemu.WORK = work
        run_qemu.main("q8")
        actual = read(build/"result.json")
        archive = RESULTS/f"{STEM}-uart.zip"
        with ZipFile(archive,"x",ZIP_DEFLATED) as z:
            z.write(build/"uart.log","uart.log")
    for st in streams.values():
        free, notices = 0, []
        for idx in st["windows"]:
            case, src = actual["records"][idx], rows[idx]
            assert case["proposal"] == src["proposal"] and case["confirmed"] == src["confirmed"]
            note = service(src,case,free)
            free = note["worker_free_us"]
            case.update(window_pcm_sha256=src["window_pcm_sha256"], start=src["start"], end=src["end"], schedule=note)
            if note["notify_us"] is not None:
                notices.append(us_to_sample(note["notify_us"]))
                assert note["notify_us"]-sample_to_us(case["proposal"]) == case["modeled_confirm_us"] + note["queue_us"]
        st["events"] = host.base.scoring.merged(st["c161"],notices)
    save(RESULTS/f"{STEM}.json",dict(**actual,streams=streams,sources=sources,uart_archive_sha256=sha(archive),
         baseline_sha256=sha(BASELINE), script_sha256=sha(Path(__file__)),
         policy="Name-contrast Q8 on all 49 continuous conversation proposals, all 12 noise-only proposals, "
         "and the one confirmed long-background proposal. The background window is isolated. One FIFO worker "
         "per stream with actual P4 QEMU block costs and existing C161 notification clocks. Excludes audio "
         "capture, proposer execution and board workload."))


if __name__ == "__main__":
    main()
