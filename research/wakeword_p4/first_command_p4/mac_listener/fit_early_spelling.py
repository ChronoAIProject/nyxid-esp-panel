"""Calibrate one optional early spelling check while retaining the standard 360 ms fallback."""
from pathlib import Path

import numpy as np

import replay_proposal_deadlines as replay
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from train_c161_mined_room_word import negative_streams

prior,base,gap = replay.prior,replay.base,replay.gap
STEM = "4090-c161-early-spelling"
FIT = RESULTS/(STEM+"-fit.json")
OFFSETS = (0,2880)
WAIT = base.scoring.word.WAIT


class Gates:
    def __init__(self):
        self.native = prior.fit.native_frontend()
        self.student, self.matcher = prior.model(),prior.Matcher()
        self.config = read(gap.FIT)
        spec = read(RESULTS/"4090-c161-positive-readout-exports.json")["models"]["trigger"]
        assert sha(RESULTS/spec["file"])==spec["sha256"]
        self.five=base.poll.Model(RESULTS/spec["file"])
        stem = replay.fit.music.STEM
        file = RESULTS/f"{stem}-v1.tflite"
        assert sha(file)==read(RESULTS/f"{stem}-export-v1.json")["sha256"]
        self.front=REF_KERNEL(model_path=str(file))
        self.front.allocate_tensors()
        self.ids = dict(proposer=sha(file),five=spec["sha256"],character=read(prior.fit.EXPORT)["models"]["student"]["sha256"],
                        config=sha(gap.FIT))

    def windows(self,raw,ends):
        checks=replay.verifier(raw,ends,self.student,self.matcher,self.config)
        for end,c in checks.items():
            patch=np.rint(prior.fit.room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
            c["nyx_code"]=int(self.five.invoke(patch)[0])
            c["passes"]=c["accepted"] and c["nyx_code"]>=0
        return checks

    def proposals(self,bands,clock):
        trace=base.infer(self.front,bands)
        runtime=base.scoring.word
        return [t for t in runtime.times(runtime.decode(trace,clock)) if t>0]


def training_streams():
    inventory={r["key"]:r for r in read(RESULTS/(replay.fit.music.word.STEM+"-training-v1.json"))["training"]}
    render=replay.fit.music.mining.room_render()
    for src,wave in replay.fit.music.word.sources():
        old=inventory.pop(src["key"])
        assert old["pcm_sha256"]==base.poll.digest(wave) and old["positive"]==src["positive"]
        dry=np.pad(wave,(32000,16000))
        wet,setting=render(src,dry,32000)
        for name,pcm in (("dry",dry),("room",wet)):
            if name=="room" and np.array_equal(wet,dry):
                continue
            yield dict(key=src["key"],voice=src["voice"],text=src["text"],mode=name,positive=src["positive"],
                       tail=32000+src["end"] if src["positive"] else None,
                       source_sha256=old["pcm_sha256"],start=32000,stop=32000+len(wave)),pcm
    assert not inventory
    for src,wave in negative_streams():
        if src["corpus"]!="LibriSpeech train-clean-100":
            break
        yield dict(key=src["key"],text=src["text"],mode="speech",positive=False,tail=None,
                   source_sha256=src["sha256"],start=16000,stop=16000+len(wave)),np.pad(wave,(16000,16000))


def calibrate():
    gates=Gates()
    rows=[]
    for src,pcm in training_streams():
        raw,bands,clock=base.native_inputs(gates.native,pcm)
        ts=[t for t in gates.proposals(bands,clock) if t<=len(pcm)]
        ends=sorted({t+d for t in ts for d in (*OFFSETS,WAIT) if t+d<=len(pcm)})
        checks=gates.windows(raw,ends)
        entries=[]
        for t in ts:
            # Positive timing credit is restricted to the labelled word. Negative streams
            # count every proposal, including prefix, tail and any later spoken command.
            near=src["positive"] and src["start"]<=t<=src["stop"] and abs(t-src["tail"])<=3840
            entries.append(dict(t=t,near=near,checks={str(d):checks[t+d] for d in (*OFFSETS,WAIT) if t+d in checks}))
        rows.append(dict(**src,pcm_sha256=base.poll.digest(pcm),proposals=entries))
    fits=[]
    for delay in OFFSETS:
        # Maintain the late rule's negative-proposal rejection even on negatives whose
        # other proposal was already a false wake. Include the late-rejected positives'
        # off-name proposals as nuisance events.
        bad=[]
        for r in rows:
            for p in r["proposals"]:
                e=p["checks"].get(str(delay),{})
                late=p["checks"].get(str(WAIT),{})
                if e.get("passes") and not late.get("passes") and not p["near"]:
                    bad.append(e["score"])
        threshold=max(gates.config["threshold"],float(np.nextafter(max(bad),np.inf))) if bad else gates.config["threshold"]
        admitted=before=rescued=0
        for r in rows:
            successes=[]
            for p in r["proposals"]:
                e=p["checks"].get(str(delay),{})
                if p["near"] and e.get("passes") and e["score"]>=threshold:
                    successes.append((p["t"]+delay,p["checks"].get(str(WAIT),{}).get("passes",False)))
            if successes:
                admitted+=1
                before+=min(t for t,_ in successes)<r["tail"]
                rescued+=not any(p["near"] and p["checks"].get(str(WAIT),{}).get("passes") for p in r["proposals"])
        fits.append(dict(delay=delay,threshold=threshold,nuisance_early_only=len(bad),admitted=admitted,
                         before_tail=before,rescued=rescued))
    choice=max(fits,key=lambda f:(f["before_tail"],f["admitted"],-f["delay"]))
    save(FIT,dict(models=gates.ids,records=rows,options=fits,selected=choice,script_sha256=sha(Path(__file__)),
         policy="Original proposer and frozen five-class/spelling gate, complete primed training-only audio. "
                "Try a single check at proposal or 180 ms later, preserve the 360 ms fallback. Set the early "
                "spelling cutoff above all additional off-name or negative proposal scores that the fallback "
                "would reject. Choose most positive views admitted before annotated name end, then total "
                "positive views, then shorter delay. All clocks in 16 kHz samples; no development use to fit."))


if __name__=="__main__":
    calibrate()
