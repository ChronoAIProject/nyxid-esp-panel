"""Set alternative part thresholds from conversation calibration and replay the unchanged integer models."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import score_ordered_acoustic_parts as replay

fit = replay.fit
CAL = fit.RESULTS / (fit.STEM + "-conversation-calibration.json")


def calibrate():
    report = fit.read(fit.FIT)
    rows = [r for r in report["records"] if r["split"]=="calibration" and not r["positive"]
            and not r["text"].casefold().startswith("hey ")]
    cutoffs = {}
    for name in fit.GAPS:
        maximum = max(r["peak"][name] for r in rows)
        cutoffs[name] = dict(value=maximum+1, maximum_keys=[r["key"] for r in rows if r["peak"][name]==maximum],
                             calibration_positives=sum(r["peak"][name]>maximum for r in report["records"]
                                                      if r["positive"] and r["split"]=="calibration"))
    data = dict(fit_sha256=fit.sha(fit.FIT),rows=[r["key"] for r in rows],cutoffs=cutoffs,
                policy="Follow-up after the strict comparison. Set each integer cutoff one above the largest whole "
                       "stream negative on calibration voices excluding elicited Hey-addresses. Common-word sentences "
                       "including next, mix and six remain in calibration. Compare both arms and simultaneous "
                       "two-frame evidence in both arms. Use all development controls including other names. "
                       "No operating point is selected from development.")
    if CAL.exists():
        assert fit.read(CAL)==data
    else:
        fit.save(CAL,data)


def score(mode):
    base = fit.RESULTS / (fit.STEM+"-"+mode+".json")
    history = fit.read(base)
    thresholds = fit.read(CAL)
    assert fit.sha(fit.FIT) == thresholds["fit_sha256"] == history["sources"][fit.FIT.name]
    native = fit.frontend.native_frontend()
    models = {}
    for n,m in history["models"].items():
        path = fit.RESULTS/m["file"]
        assert fit.sha(path)==m["sha256"]
        with np.load(path) as z:
            models[n] = (dict(z),m["gaps"],thresholds["cutoffs"][n]["value"])
    records,summary,sources = [],defaultdict(lambda:defaultdict(Counter)),{}
    for old,(row,src,c161,pcm) in zip(history["records"],replay.corpus(mode,sources),strict=True):
        assert (old["key"],old["group"],old["pcm_sha256"],old["c161"]) == (row["key"],row["group"],src["pcm_sha256"],c161)
        traces,clock = {},None
        if any(old["candidates"][n]["peak"] >= cut for n,(_,_,cut) in models.items()):
            if pcm is None:
                pcm,rate = sf.read(src.get("wav_path") or src["path"],dtype="int16")
                assert rate==16000
            assert sha256(pcm.tobytes()).hexdigest()==src["pcm_sha256"]
            bands,clock = replay.native_features(native,pcm)
            for n,(weights,gaps,cut) in models.items():
                scores = fit.integer_scores(bands,weights,gaps)
                assert sha256(scores.tobytes()).hexdigest()==old["candidates"][n]["score_sha256"]
                traces[n] = scores-cut
            traces["both"] = np.minimum(traces["fixed"],traces["variable"])
        row.update(pcm_sha256=src["pcm_sha256"],samples=old["samples"],c161=c161,events={},union={})
        for name in (*models,"both"):
            hits = replay.events(traces[name],clock,0) if traces else []
            row["events"][name] = hits
            combined = [e["sample"] for e in replay.choose(c161,hits)]
            row["union"][name] = combined
            for model,ts in ((name,hits),("c161+"+name,combined)):
                ctr = summary[row["group"]+"."+row["role"]][model]
                ctr.update(cases=1,opened=bool(ts),events=len(ts),gained=bool(ts) and not c161,
                           lost=bool(c161) and not ts,samples=old["samples"])
                if "command_at" in row:
                    first = (ts[0]+255)//256*256 if ts else None
                    ctr.update(timely=first is not None and row["start"]<=first<row["command_at"])
        records.append(row)
        if mode=="background" and len(records)%10==0:
            print("background",len(records),flush=True)
    dest = fit.RESULTS / (fit.STEM+"-conversation-"+mode+".json")
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        p = Path(tmp)/dest.name
        fit.save(p,dict(records=records,summary=summary,sources={p.name:fit.sha(p) for p in
                    (CAL,base,Path(__file__),Path(replay.__file__),fit.FIT)}))
        p.replace(dest)


if __name__=="__main__":
    action, = sys.argv[1:]
    if action=="calibrate":
        calibrate()
    else:
        assert action in ("speech","background")
        score(action)
