"""Replay every proposed five-class window from the room study through the P4 kernel."""
import argparse
import hashlib
from pathlib import Path
import re
import sys

import numpy as np

APP=Path(__file__).resolve().parent
sys.path.insert(0,str(APP.parent/"experiments"))
import check_c161_polling_proposals as check
import score_c161_polling_rooms as poll
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import boot
from run_c161_coded_cue_p4_v5 import QEMU

GROUPS=("house","fresh","daily","context","bot")


def main(build,dest):
    dest.mkdir(parents=True,exist_ok=True)
    model=RESULTS/read(RESULTS/(poll.fit.STEM+"-exports.json"))["models"]["room"]["file"]
    images=read(build/"flasher_args.json")["flash_files"]
    assert sum((build/n).read_bytes().count(model.read_bytes()) for n in images.values())==1
    assert read(build/"config/sdkconfig.json")["NN_ANSI_C"]
    lookup=poll.Model(model).table
    blocks,rows=[],[]
    sources={}
    for group in GROUPS:
        report=RESULTS/(check.STEM+"-"+group+".json")
        sources[group]=sha(report)
        if group in ("fresh","house"):
            plan=read(poll.fresh.PLAN if group=="fresh" else poll.HOME_SCENES)["records"]
            waves=((r["index"],poll.load_wav(plan[r["index"]])) for r in read(report)["records"])
        else:
            waves=((i,pcm) for i,_,pcm,_ in poll.live_sources(group))
        for (i,pcm),r in zip(waves,read(report)["records"],strict=True):
            assert r["index"]==i and poll.digest(pcm)==r["pcm_sha256"]
            if not r["checks"]:
                continue
            bands=poll.fit.raw_features(pcm)
            for event in r["checks"]:
                at=event["available"]
                patch=lookup[poll.fit.window(bands,at)]
                y=np.array(event["codes"],dtype=np.int8)
                blocks.append(patch.tobytes()+y.tobytes())
                rows.append(dict(group=group,record=i,at=at,host_codes=y.tolist(),input_sha256=poll.digest(patch)))
    assert len(rows)==len(blocks) and 0<len(rows)<=300
    data=len(blocks).to_bytes(4,"little")+b"".join(blocks)
    log=dest/(check.STEM+"-p4-uart.log")
    boot(build,log,poll.fit.WORK/"p4-flash.bin",b"ONSET_DONE",(2<<20,data))
    text=log.read_text()
    assert "ONSET_FC per_channel" in text and "ONSET_FAIL" not in text
    size,init=map(int,re.search(r"ONSET_READY arena=(\d+) init_us=(\d+)",text).groups())
    count,heap=map(int,re.search(r"ONSET_END count=(\d+) min_internal=(\d+)",text).groups())
    results=[list(map(int,s.split())) for s in re.findall(r"ONSET_ROW ([0-9 -]+)",text)]
    assert len(rows)==count
    for i,(r,out) in enumerate(zip(rows,results,strict=True)):
        assert out[0]==i and out[1:6]==r["host_codes"] and out[6]>0
        r.update(device_codes=out[1:6],invoke_us=out[6])
    save(dest/(check.STEM+"-p4.json"),dict(records=rows,arena_bytes=size,init_us=init,min_internal_free=heap,
         qemu_sha256=sha(QEMU),model_sha256=sha(model),fixture_sha256=hashlib.sha256(data).hexdigest(),
         uart_sha256=sha(log),host_reports=sources,images={n:sha(build/n) for n in images.values()},
         scope="All completed proposal confirmations from five development groups. Host features, P4 QEMU ANSI-C convolution and per-channel FC. Invoke time excludes frontend, word detector, transfer and concurrent panel tasks."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("build",type=Path)
    p.add_argument("--out",type=Path,default=poll.fit.WORK/"p4-results")
    args=p.parse_args()
    main(args.build,args.out)
