"""Train an integer linear early verifier on frozen five-class and spelling outputs."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import time_calibrated_echo_names as timing

ch = timing.ch
training = timing.timing.study.training
FIT = ch.RESULTS / "4090-c161-early-echo-gate-fit.json"
COEFF = ch.RESULTS / "4090-c161-early-echo-gate.json"
FEATURES = ["target_times_16", "rival_times_16", "blank_frames"] + [f"five_{i}" for i in range(5)]


def features(w, codes):
    return [round(w["target_value"]*16), round(w["rival_value"]*16), w["gap"], *codes]


def collect():
    parent = ch.read(training.FIT)
    detector = timing.spoken.natural.Detector()
    assert parent["front_sha256"] == detector.identity["proposer"] and detector.cutoff == 34
    rows = []
    for (source, pcm), old in zip(training.sources(), parent["records"], strict=True):
        assert (source["key"], source["mode"], source["source_sha256"], timing.ch.digest(pcm)) == (
            old["key"], old["mode"], old["source_sha256"], old["pcm_sha256"])
        proposals = [t for t in old["proposals"] if 0 < t and t+timing.EARLY <= len(pcm)]
        ends = sorted({t+d for t in proposals for d in (timing.EARLY,timing.STANDARD) if t+d <= len(pcm)})
        raw = timing.timing.base.native_inputs(detector.native, pcm)[0] if ends else None
        checks = timing.spoken.natural.replay.verifier(raw, ends, detector.student, detector.matcher, detector.gap)
        for end, w in checks.items():
            if w["accepted"]:
                x = np.rint(training.room.window(raw,end).clip(0,666)*(255/666)).astype(np.uint8)
                w["five"] = detector.models["trained"].invoke(x).tolist()
        windows = []
        for t in proposals:
            end = t + timing.EARLY
            w = checks[end]
            late = checks.get(t+timing.STANDARD, {})
            target = bool(source["positive"] and source["start"] <= t <= source["stop"]
                          and source["tail"]-1920 <= end <= source["tail"]+timing.STANDARD+3840)
            x = features(w, w["five"]) if w["accepted"] else None
            windows.append(dict(proposal=t, target=target, near_features=x,
                                early_pass=w["accepted"] and w["five"][0]>=detector.cutoff,
                                late_pass=bool(late.get("accepted") and late["five"][0]>=detector.cutoff),
                                early_output=w["output_sha256"], late_output=late.get("output_sha256")))
        rows.append(dict(key=source["key"], mode=source["mode"], text=source.get("text"), positive=source["positive"],
                         source_sha256=source["source_sha256"], pcm_sha256=old["pcm_sha256"], windows=windows,
                         split="calibrate" if int(source["source_sha256"][:8],16)%5 == 0 else "fit"))
    return rows, detector.identity


def fit():
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    rows, models = collect()
    fit_sources = {r["source_sha256"] for r in rows if r["split"]=="fit"}
    cal_sources = {r["source_sha256"] for r in rows if r["split"]=="calibrate"}
    assert not fit_sources.intersection(cal_sources)
    training_rows = [w for r in rows if r["split"]=="fit" for w in r["windows"] if w["near_features"] is not None]
    x = torch.tensor([w["near_features"] for w in training_rows], dtype=torch.float64, device="cuda")
    y = torch.tensor([w["target"] for w in training_rows], dtype=x.dtype, device="cuda")
    mean, sd = x.mean(0), x.std(0,unbiased=False).clamp_min(1)
    z = (x-mean)/sd
    weights = torch.zeros(x.shape[1],dtype=x.dtype,device="cuda",requires_grad=True)
    bias = torch.zeros((),dtype=x.dtype,device="cuda",requires_grad=True)
    opt = torch.optim.Adam((weights,bias),lr=.03)
    assert 0 < y.sum() < len(y)
    importance = torch.where(y.bool(), .5/y.sum(), .5/(len(y)-y.sum()))
    losses = []
    for step in range(300):
        preds = z @ weights + bias
        loss = (F.binary_cross_entropy_with_logits(preds,y,reduction="none")*importance).sum() + .01*weights.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,99,299):
            losses.append(float(loss.detach()))
    w = (weights/sd).detach()
    b = (bias-mean@w).detach()
    multiplier = 4096/float(w.abs().max())
    coeff = torch.round(multiplier*w).to(torch.int64).tolist()
    boundary = int(np.ceil(float(-b)*multiplier))
    negative = [w["near_features"] for r in rows if r["split"]=="calibrate" for w in r["windows"]
                if w["early_pass"] and not w["target"] and not w["late_pass"]]
    cutoff = max([boundary, *(1+int(np.dot(coeff,x)) for x in negative)])
    scalar = max([models["cutoff"], *(x[3]+1 for x in negative)])
    counts = Counter()
    for r in rows:
        for w in r["windows"]:
            if w["near_features"] is None:
                continue
            x = w["near_features"]
            score = int(np.dot(coeff,x))
            assert -(1<<31) <= score < (1<<31)
            assert not (r["split"]=="calibrate" and not w["target"] and not w["late_pass"]
                        and w["early_pass"] and (score>=cutoff or x[3]>=scalar))
            for mode, admitted in (("base",True),("linear",score>=cutoff),("scalar",x[3]>=scalar)):
                if w["early_pass"] and admitted:
                    counts[".".join((r["split"],mode,"target" if w["target"] else "other"))] += 1
    ch.save(FIT, dict(records=rows, models=models, previous_training_sha256=ch.sha(training.FIT), features=FEATURES,
            fit_streams=len(fit_sources), calibration_streams=len(cal_sources), fit_windows=len(training_rows),
            fit_targets=int(y.sum()), calibration_early_only_nuisance=len(negative), losses=losses,
            counts=counts, gpu=device, steps=300, script_sha256=ch.sha(Path(__file__)),
            policy="Group training streams by original audio SHA256, modulo five for separate calibration. "
            "Use the frozen 180 ms spelling-gated windows for a class-balanced logistic fit. Label target "
            "windows around the marked name end; off-name proposals are nuisance. Fold feature scaling and "
            "round coefficients, then gate only the original early positives. Set thresholds on the calibration "
            "split to prevent extra hits on any nuisance window rejected at 360 ms. Fallback remains frozen."))
    ch.save(COEFF,dict(features=FEATURES,weights=coeff,linear_cutoff=cutoff,uncalibrated_cutoff=boundary,
                       scalar_nyx_cutoff=scalar, fit_sha256=ch.sha(FIT),models=models))


if __name__ == "__main__":
    fit()
