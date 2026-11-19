"""Train a late-window Nick veto from one or two character snapshots with unchanged early acceptance."""
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

import check_early_gate_voices as voices

ch, timing = voices.ch, voices.timing
FIT = ch.RESULTS/"4090-c161-late-nick-gate-fit.json"


def spelling(w):
    return [round(16*w["target_value"]),round(16*w["rival_value"]),int(w["rival"].endswith("NICK"))]


def features(before, late):
    return spelling(before) + spelling(late) + [late["gap"]] + late["five"]


def collect():
    training = voices.gate.fit.training
    parent = ch.read(training.FIT)
    older = ch.read(voices.gate.fit.FIT)["records"]
    net = timing.spoken.natural.Detector()
    rows=[]
    for (src,pcm),old,past in zip(training.sources(),parent["records"],older,strict=True):
        if src["mode"] not in ("dry","room"):
            continue
        nick = bool(re.match(r"^(hey|hi) nick\b",src["text"],re.I))
        if not (src["positive"] or nick):
            continue
        assert src["key"]==past["key"]==old["key"] and ch.digest(pcm)==past["pcm_sha256"]==old["pcm_sha256"]
        ts = [p["t"] for p in old["training"]]
        if not ts:
            continue
        raw = timing.timing.base.native_inputs(net.native,pcm)[0]
        ends = sorted({t+d for t in ts for d in (timing.EARLY,timing.STANDARD)})
        checks = timing.spoken.natural.replay.verifier(raw,ends,net.student,net.matcher,net.gap)
        for t in ts:
            a,b=checks[t+timing.EARLY], checks[t+timing.STANDARD]
            if not b["accepted"]:
                continue
            patch=np.rint(training.room.window(raw,t+timing.STANDARD).clip(0,666)*(255/666)).astype(np.uint8)
            b["five"]=net.models["trained"].invoke(patch).tolist()
            previous=next(p for p in past["windows"] if p["proposal"]==t)
            assert previous["early_output"]==a["output_sha256"] and previous["late_output"]==b["output_sha256"]
            assert previous["late_pass"]==(b["five"][0]>=net.cutoff)
            rows.append(dict(key=src["key"],voice=src["voice"],mode=src["mode"],split=past["split"],
                        source_sha256=src["source_sha256"],target=src["positive"],t=t,features=features(a,b),
                        late_pass=previous["late_pass"]))
    return rows,net.identity


def train(rows, start):
    learn=[r for r in rows if r["split"]=="fit"]
    cal=[r for r in rows if r["split"]=="calibrate" and r["late_pass"]]
    x=torch.tensor([r["features"][start:] for r in learn],dtype=torch.float64,device="cuda")
    y=torch.tensor([r["target"] for r in learn],dtype=x.dtype,device=x.device)
    center,scale=x.mean(0),x.std(0,unbiased=False).clamp_min(1)
    z=(x-center)/scale
    w=torch.zeros(x.shape[1],dtype=x.dtype,device=x.device,requires_grad=True)
    b=torch.zeros((),dtype=x.dtype,device=x.device,requires_grad=True)
    loss_w=torch.where(y.bool(),.5/y.sum(),.5/(len(y)-y.sum()))
    opt=torch.optim.Adam((w,b),lr=.03)
    for _ in range(300):
        loss=(F.binary_cross_entropy_with_logits(z@w+b,y,reduction="none")*loss_w).sum()+.01*w.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Only the ordering is used. Set the veto boundary to the weakest admitted
    # calibration positive. The intercept and folded feature means cancel.
    w=(w/scale).detach()
    q=torch.round(4096*w/w.abs().max()).to(torch.int64).tolist()
    def value(r):
        return sum(a*b for a,b in zip(q,r["features"][start:],strict=True))
    cutoff=min(value(r) for r in cal if r["target"])
    return dict(weights=q,offset=start,cutoff=cutoff,fit_windows=len(learn),fit_targets=int(y.sum()),
                calibrated_targets=sum(r["target"] for r in cal),calibrated_nick=sum(not r["target"] for r in cal),
                calibrated_nick_pass=sum(not r["target"] and value(r)>=cutoff for r in cal))


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    rows,models=collect()
    assert {r["source_sha256"] for r in rows if r["split"]=="fit"}.isdisjoint(
        r["source_sha256"] for r in rows if r["split"]=="calibrate")
    models_fit={k:train(rows,s) for k,s in (("late",3),("paired",0))}
    ch.save(FIT,dict(records=rows,models=models,heads=models_fit,device=torch.cuda.get_device_name(),
            training_sha256=ch.sha(voices.gate.fit.training.FIT),partition_sha256=ch.sha(voices.gate.fit.FIT),
            script_sha256=ch.sha(Path(__file__)),policy="Binary Nyx versus exact Hey/Hi Nick training sources. "
            "Use admitted 360 ms spelling windows from prior training offsets. Fit a class-balanced logistic head on the "
            "late spelling, word gap and five-class codes, or also the earlier target/rival scores and rival Nick flag. "
            "Choose one integer cutoff preserving all previously admitted target calibration windows from separate "
            "source audio hashes. When replayed only veto the original 360 ms fallback, never a 180 ms or direct C161 hit."))


if __name__=="__main__":
    main()
