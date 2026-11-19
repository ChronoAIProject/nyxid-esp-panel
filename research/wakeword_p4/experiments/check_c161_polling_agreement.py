"""Test agreement of the two existing compact word detectors before five-class confirmation."""
import numpy as np

import check_c161_polling_proposals as check
import score_c161_polling_rooms as poll
from check_c161_address_grammar import merged
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_room_word_background import infer
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from score_c161_piper_spellings import annotate
from train_c161_room_word import ORIGINAL
from train_c161_word_window_v1 import bands

STEM=check.STEM+"-agreement"


def main():
    models=[]
    for name in (ORIGINAL,check.WORD):
        p=RESULTS/(name+"-v1.tflite")
        assert sha(p)==read(RESULTS/(name+"-export-v1.json"))["sha256"]
        vm=REF_KERNEL(model_path=str(p))
        vm.allocate_tensors()
        models.append(vm)
    reader,wh=load_reader()
    all_rows={}
    sources={}
    for group in ("house","fresh","daily","context","bot","background"):
        path=RESULTS/(check.STEM+"-"+group+".json")
        src=read(path)["records"]
        sources[group]=sha(path)
        rows=[]
        if group=="background":
            original=read(RESULTS/(check.WORD+"-background.json"))
            archive=RESULTS/(check.WORD+"-background-scores.npz")
            assert sha(archive)==original["scores_sha256"]
            with np.load(archive) as z:
                for key,r in src.items():
                    c,h=z[key+"_clean"],z[key+"_mined"]
                    support=(3*np.arange(1,len(h)+1,dtype=np.int64)+2)*160
                    clock=np.minimum(((support+479)//480)*480,original["records"][key]["samples"]+poll.fit.PRIME)-poll.fit.PRIME
                    assert poll.digest(clock)==original["records"][key]["clock_sha256"]
                    checks=[]
                    for event in r["checks"]:
                        p=event["proposal"]
                        tick=int(np.searchsorted(clock,p))
                        assert clock[tick]==p and h[tick-4:tick+1].sum()>=1175
                        total=int(np.minimum(c[tick-4:tick+1],h[tick-4:tick+1]).sum())
                        checks.append(dict(at=event["available"],sum_min=total,nyx=event["codes"][0]))
                    rows.append(dict(key=key,checks=checks,accepted=[s["at"] for s in checks if s["nyx"]>=0 and s["sum_min"]>=1175]))
        else:
            if group in ("fresh","house"):
                plan=read(poll.fresh.PLAN if group=="fresh" else poll.HOME_SCENES)["records"]
                data=((r["index"],plan[r["index"]],poll.load_wav(plan[r["index"]])) for r in src)
            else:
                data=((i,m,p) for i,m,p,_ in poll.live_sources(group))
            prior=read(RESULTS/(poll.fit.STEM+"-"+group+".json"))["records"]
            for (i,meta,pcm),r,baseline in zip(data,src,prior,strict=True):
                assert i==r["index"]==baseline["index"] and poll.digest(pcm)==r["pcm_sha256"]==baseline["pcm_sha256"]
                c161=baseline["routes"]["c161"]
                checks=[]
                if r["checks"]:
                    x,clock=bands(pcm)
                    for event in r["checks"]:
                        p=event["proposal"]
                        tick=int(np.searchsorted(clock,p))
                        assert clock[tick]==p
                        indices=range(tick-4,tick+1)
                        c,h=[infer(vm,x,indices) for vm in models]
                        assert h.sum()>=1175
                        checks.append(dict(at=event["available"],sum_min=int(np.minimum(c,h).sum()),nyx=event["codes"][0]))
                accepted=[s["at"] for s in checks if s["sum_min"]>=1175 and s["nyx"]>=0]
                routes=dict(agreement=merged(c161,accepted))
                if group=="fresh":
                    result=annotate(meta,pcm,routes,[],[],reader)
                    for row in result["rows"]:
                        del row["native_proposals"],row["expanded_proposals"]
                else:
                    labels=[]
                    outside=set(routes["agreement"])
                    for span in r["rows"]:
                        hits=[v for v in routes["agreement"] if span["start"]<=v<span["end"]]
                        outside.difference_update(hits)
                        labels.append(dict(address=span["address"],start=span["start"],end=span["end"],routes=dict(agreement=dict(hits=hits))))
                    result=dict(rows=labels,outside=dict(agreement=sorted(outside)))
                rows.append(dict(index=i,pcm_sha256=r["pcm_sha256"],meta=r["meta"],checks=checks,routes=routes,**result))
        all_rows[group]=rows
        print(group,len(rows),flush=True)
    save(RESULTS/(STEM+".json"),dict(records=all_rows,source_reports=sources,asr=wh,
       policy="At the frozen hard-negative word proposal, require sum of the five per-tick minimum UINT8 codes from clean and hard models >=1175. Only then invoke the same five-class room confirmation after 360 ms. C161 stays independent. This diagnostic was added after the development background review."))


if __name__=="__main__":
    main()
