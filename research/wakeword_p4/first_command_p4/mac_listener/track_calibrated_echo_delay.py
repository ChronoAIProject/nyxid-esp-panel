"""Challenge online playback-reference alignment on already used connected calls and competing names."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import correlate
import torch

import fit_early_echo_gate as fitted
import score_early_echo_gate as gated

timing, ch = fitted.timing, fitted.ch
OUT = ch.RESULTS / "4090-c161-calibrated-reference-delay.json"
HOP = timing.names.HOP
PAST = 8 * HOP
FAULTS = ("stable","late_16ms","early_16ms","drift_200ppm","reference_lost")


def electrical(ref, start, mode):
    if mode == "stable":
        return ref.copy()
    out = ref.copy()
    indices = np.arange(start,len(ref))
    if mode == "reference_lost":
        out[start:] = 0
    elif mode == "drift_200ppm":
        out[start:] = np.rint(np.interp(indices-(indices-start)*.0002,np.arange(len(ref)),ref)).astype("<i2")
    else:
        shift = HOP if mode=="late_16ms" else -HOP
        out[start:] = np.interp(indices-shift,np.arange(len(ref)),ref,left=0,right=0).astype("<i2")
    return out


def aligned(mic, prediction, cutoff):
    # Output block [at,at+HOP) becomes available at at+2*HOP. The delay search
    # uses only earlier mic samples. Its reference support ends no later than at+HOP.
    out = np.zeros(len(mic)-HOP,dtype="<i2")
    shift = 0
    changes = []
    yp = prediction.astype(np.float64)
    sums = np.r_[0., np.cumsum(yp)]
    squares = np.r_[0., np.cumsum(yp*yp)]
    for at in range(cutoff, len(out), HOP):
        first = at-PAST
        mic_hist = mic[first:at].astype(np.float64)
        mic_hist -= mic_hist.mean()
        region = yp[first-HOP:at+HOP]
        scale = squares[first-HOP+PAST:at+HOP+1] - squares[first-HOP:at+HOP+1-PAST]
        sums_y = sums[first-HOP+PAST:at+HOP+1] - sums[first-HOP:at+HOP+1-PAST]
        energy = np.maximum(0., scale - sums_y*sums_y/PAST)
        norm = np.sqrt(np.maximum(1., energy*(mic_hist@mic_hist)))
        corr = correlate(region,mic_hist,mode="valid",method="fft") / norm
        assert len(corr) == 2*HOP+1
        best = int(corr.argmax())
        if (energy[best] >= PAST*64**2 and corr[best] >= .55
                and corr[best] >= corr[shift+HOP]+.10):
            shift = best-HOP
            changes.append(dict(at=at,shift=shift,rho=float(corr[best])))
        n = min(HOP,len(out)-at)
        out[at:at+n] = np.rint(mic[at:at+n]-yp[at+shift:at+shift+n]).clip(-32768,32767).astype("<i2")
    return out, changes


def sources(parent):
    refs = {}
    for meta, self_mic, ref in timing.spoken.extra_cases():
        if meta["mode"]!="self_only" or meta["bot"] in refs:
            continue
        refs[meta["bot"]] = ref
        yield dict(role="self",voice=meta["bot"],bot=meta["bot"],start=meta["cue_at"],
                   end=len(ref)-HOP,command_at=None,command=None,db=None), self_mic, ref, None
    for row,mic,clean in timing.signals(parent):
        matched = (row["role"]=="Hey Nyx" and row["command"]=="can you unlock the side door?")
        conflict = (row["role"]=="Hey Nick" and row["command"]=="turn down the hall light.")
        if matched or conflict:
            yield row,mic,refs[row["bot"]],clean


def run():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    net = timing.spoken.natural.Detector()
    parent = ch.read(timing.names.OUT)
    spec = ch.read(fitted.COEFF)
    assert parent["models"] == net.identity == spec["models"]
    cutoff = parent["cutoff_samples"]
    assert ch.sha(timing.names.COEFF) == parent["coeff_sha256"]
    char_us, five_us = parent["char_us"], parent["five_us"]
    reader, asr = ch.load_reader()
    assert asr == parent["asr"]
    raw_events, signals, readings, records = {}, {}, {}, []
    totals = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    with np.load(timing.names.COEFF,allow_pickle=False) as filters:
        for row, mic, ref, known in sources(parent):
            key = ch.digest(mic)
            if key not in raw_events:
                raw_events[key] = net.score(mic)["c161"]
            if known is not None:
                assert raw_events[key] == row["stages"]["raw"]["c161"]
            for fault in FAULTS:
                electrical_ref = electrical(ref,row["start"],fault)
                prediction = timing.names.estimate(electrical_ref,filters[row["bot"]])
                stable = np.rint(mic[:-HOP]-prediction[:-HOP].astype(float)).clip(-32768,32767).astype("<i2")
                stable[:cutoff] = 0
                tracked, changes = aligned(mic,prediction,cutoff)
                if fault=="stable" and known is not None:
                    np.testing.assert_array_equal(known[:-HOP],stable)
                obs = dict(role=row["role"],voice=row["voice"],db=row["db"],command=row["command"],
                           command_at=row["command_at"],start=row["start"],end=row["end"],fault=fault,
                           mic_sha256=key,reference_sha256=ch.digest(electrical_ref),changes=changes,routes={})
                for name,pcm in (("static",stable),("tracked",tracked)):
                    digest = ch.digest(pcm)
                    if digest not in signals:
                        # Frozen proposer and C161 clocks; generate the additional 180 ms checks.
                        raw = timing.timing.base.native_inputs(net.native,pcm)
                        word = timing.timing.base.scoring.word
                        prop = [t for t in word.times(word.decode(timing.timing.base.infer(net.front,raw[1]),raw[2])) if 0<t<=len(pcm)]
                        trace = net.score(pcm,[t+timing.EARLY for t in prop if t+timing.EARLY<=len(pcm)])
                        assert trace["proposals"] == prop
                        signals[digest] = trace
                    trace = signals[digest]
                    windows,features = {},{}
                    for end,w in trace["checks"].items():
                        assert end >= cutoff
                        windows[end] = dict(id=end,service_us=char_us+(five_us if w["accepted"] else 0),passes=w["pass"])
                        if w["accepted"]:
                            features[end] = fitted.features(w,w["five"])
                    # Translate the residual, its verifier windows and its independent C161 to input-arrival time.
                    direct = [timing.block(t) for t in raw_events[key]]
                    direct.extend(timing.block(t)+HOP for t in trace["c161"] if t>=cutoff)
                    shifted = dict(start=row["start"],end=row["end"],command_at=row["command_at"] or 0,
                                   stages=dict(raw=dict(c161=direct),filtered=dict(c161=[],proposals=[t+HOP for t in trace["proposals"]])))
                    run = gated.routes(shifted,{t+HOP:w for t,w in windows.items()},
                                       {t+HOP:x for t,x in features.items()},spec,0)["linear"]
                    times = [t for t in run["events"] if row["start"]<=t<row["end"]]
                    result = dict(residual_sha256=digest,events=run["events"],hits=times,timely=run["before_command"],
                                  proposals=trace["proposals"])
                    count = totals[row["role"]][fault][name]
                    count.update(trials=1,hits=bool(times),timely=run["before_command"])
                    if row["role"]=="Hey Nyx" and times:
                        first = max(0,((times[0]-HOP+159)//160)*160-32000)
                        cap = pcm[first:row["end"]]
                        ck = ch.digest(cap)
                        if ck not in readings:
                            readings[ck] = timing.names.transcripts(reader,cap)
                        words = timing.names.normalized(readings[ck]["text"])
                        want = timing.names.normalized(row["command"])
                        complete = words[-len(want):] == want
                        result.update(first=first,capture_sha256=ck,complete=complete)
                        count.update(complete=complete,timely_complete=complete and run["before_command"])
                    obs["routes"][name] = result
                records.append(obs)
            print(len(records),row["role"],row["voice"],flush=True)
    assert len(records) == 130
    ch.save(OUT,dict(records=records,summary=totals,texts=readings,models=spec["models"],asr=asr,
        sources={p.name:ch.sha(p) for p in (timing.names.OUT,timing.names.COEFF,fitted.COEFF,Path(__file__))},
        policy="Identical fixed 16 ms output delay for static and tracked causal FIR residuals. Every 256 "
        "samples compare normalized correlation of the last 128 ms to the predicted echo at all +/-256 "
        "sample offsets. A new lag needs reference RMS >=64, correlation >=.55 and improvement >=.10. "
        "Faults start when the address would begin, including both reference-only controls. Independent raw "
        "C161 remains on its own input clock. Charge the archived serial verifier P95 times. Full two-second "
        "host residual PCM ASR, no Opus, filter/tracker execution or physical hardware timing."))


if __name__ == "__main__":
    run()
