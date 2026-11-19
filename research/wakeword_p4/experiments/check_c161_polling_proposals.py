"""Use the previous compact INT8 word detector to schedule one five-way check per proposal."""
import argparse
import numpy as np

import score_c161_polling_rooms as poll
from audit_c161_history_paths_v1 import decode, times
from check_c161_address_grammar import merged
from check_c161_pocketsphinx_first_command import load_reader
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from replay_c161_word_window_v1 import score
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav

WORD = "4090-c161-room-word-negatives-hard"
STEM = poll.fit.STEM+"-proposals"
WAIT = 12*480  # 360 ms following a completed smoothed word proposal


def main(group):
    file = RESULTS/(WORD+"-v1.tflite")
    assert sha(file)==read(RESULTS/(WORD+"-export-v1.json"))["sha256"]
    it=REF_KERNEL(model_path=str(file))
    it.allocate_tensors()
    inp,=it.get_input_details()
    out,=it.get_output_details()
    exports=read(RESULTS/(poll.fit.STEM+"-exports.json"))
    model=RESULTS/exports["models"]["room"]["file"]
    assert sha(model)==exports["models"]["room"]["sha256"]
    classifier=poll.Model(model)
    previous=RESULTS/(poll.fit.STEM+"-"+group+".json")
    prior=read(previous)["records"]
    if group in ("fresh","house"):
        plan=read(poll.fresh.PLAN if group=="fresh" else poll.HOME_SCENES)["records"]
        streams=((r["index"],plan[r["index"]],load_wav(plan[r["index"]]),r["routes"]["c161"]) for r in prior)
    else:
        streams=poll.live_sources(group)
    if group=="fresh":
        reader,identity=load_reader()
    rows=[]
    for (i,src,pcm,base),r in zip(streams,prior,strict=True):
        assert i==r["index"] and poll.digest(pcm)==r["pcm_sha256"] and base==r["routes"]["c161"]
        codes,clock=score(it,pcm,inp,out)
        proposals=times(decode(codes,clock))
        raw=poll.fit.raw_features(pcm)
        checks=[]
        for p in proposals:
            at=p+WAIT
            if 0<p and at<=len(pcm):
                y=classifier.invoke(poll.fit.window(raw,at)).tolist()
                checks.append(dict(proposal=p,available=at,codes=y))
        accepted=[c["available"] for c in checks if c["codes"][0]>=0]
        routes=dict(word=merged(base,proposals),verified=merged(base,accepted))
        if group=="fresh":
            result=annotate(src,pcm,routes,[],[],reader)
            for row in result["rows"]:
                del row["native_proposals"],row["expanded_proposals"]
        else:
            labels=[]
            outside={k:set(t) for k,t in routes.items()}
            for span in r["rows"]:
                row={k:v for k,v in span.items() if k!="routes"}
                row["routes"]={}
                for name,events in routes.items():
                    hits=[t for t in events if row["start"]<=t<row["end"]]
                    row["routes"][name]=dict(hits=hits)
                    outside[name].difference_update(hits)
                labels.append(row)
            result=dict(rows=labels,outside={k:sorted(v) for k,v in outside.items()})
        rows.append(dict(index=i,pcm_sha256=r["pcm_sha256"],meta=r["meta"],proposals=proposals,checks=checks,
                         routes=routes,**result))
    save(RESULTS/(STEM+"-"+group+".json"),dict(records=rows,prior_sha256=sha(previous),
         word_sha256=sha(file),five_class_sha256=sha(model),wait_samples=WAIT,
         asr=identity if group=="fresh" else None,
         policy="First five-frame word sum >=1175/1275, its native re-arm and cooldown, then one five-class window 360 ms later. Nyx code >=0. Original C161 independently accepts. Time is last input sample available."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("group",choices=("house","fresh","daily","context","bot"))
    main(p.parse_args().group)
