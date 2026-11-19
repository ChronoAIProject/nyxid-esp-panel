"""Test whether a separately clocked local playback copy can cover a dead electrical reference."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import torch

import track_calibrated_echo_delay as prior

timing, ch = prior.timing, prior.ch
HOP = timing.names.HOP
OUT = ch.RESULTS/"4090-c161-render-copy-dropout.json"
COEFF = OUT.with_suffix(".npz")


def writer(ref, start, extra):
    # A queued, pre-existing PCM file can be written ahead of physical playout. The writer sees
    # only samples presented at the current write clock. Test jumps in that clock separately.
    indices = np.arange(len(ref)) + 640
    indices[start:] += extra
    return np.interp(indices,np.arange(len(ref)),ref,left=0,right=0).astype("<i2")


def fallback(mic, adc, physical, digital, cutoff):
    direct = np.rint(mic.astype(float)-physical).clip(-32768,32767).astype("<i2")
    direct[:cutoff] = 0
    copy, gated = direct.copy(), direct.copy()
    choices = {"copy":[], "guarded":[]}
    delta = digital.astype(float)-physical
    residual = mic.astype(float)-physical
    for at in range(cutoff,len(mic)-HOP,HOP):
        end = at+HOP
        # Detect a digital zero run across two full ADC blocks. During normal quiet
        # both predictions can be silent, hence the additional minimum copy energy.
        if adc[end-2*HOP:end].any():
            continue
        first = end-8*HOP
        d = delta[first:end]
        energy = d@d
        if energy < len(d)*64**2:
            continue
        block = np.rint(mic[at:end].astype(float)-digital[at:end]).clip(-32768,32767).astype("<i2")
        copy[at:end] = block
        choices["copy"].append(at)
        if 2*(residual[first:end]@d) > 1.25*energy:
            gated[at:end] = block
            choices["guarded"].append(at)
    return {"electrical":direct[:-HOP],"copy":copy[:-HOP],"guarded":gated[:-HOP]}, choices


def score(net, mic_events, pcm, row, spec, costs):
    base = timing.timing.base
    _,bands,clock = base.native_inputs(net.native,pcm)
    word = base.scoring.word
    props = [t for t in word.times(word.decode(base.infer(net.front,bands),clock)) if 0<t<=len(pcm)]
    trace = net.score(pcm,[t+timing.EARLY for t in props if t+timing.EARLY<=len(pcm)])
    assert props==trace["proposals"]
    windows,features = {},{}
    for t,w in trace["checks"].items():
        assert t>=costs["cutoff_samples"]
        time = t+HOP
        windows[time] = dict(id=t,service_us=costs["char_us"]+(costs["five_us"] if w["accepted"] else 0),passes=w["pass"])
        if w["accepted"]:
            features[time] = prior.fitted.features(w,w["five"])
    direct = [timing.block(t) for t in mic_events]
    direct.extend(timing.block(t)+HOP for t in trace["c161"] if t>=costs["cutoff_samples"])
    state = dict(start=row["start"],end=row["end"],command_at=row["command_at"] or 0,
                 stages=dict(raw=dict(c161=direct),filtered=dict(c161=[],proposals=[t+HOP for t in props])))
    return prior.gated.routes(state,windows,features,spec,0)["linear"]["events"]


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    parent, old = ch.read(timing.names.OUT), ch.read(prior.OUT)
    lost = {r["mic_sha256"]:r for r in old["records"] if r["fault"]=="reference_lost"}
    net = timing.spoken.natural.Detector()
    spec = ch.read(prior.fitted.COEFF)
    assert net.identity==parent["models"]==spec["models"]
    reader,asr = ch.load_reader()
    assert asr==old["asr"]
    inputs = list(prior.sources(parent))
    plan = ch.read(timing.spoken.connected.PLAN)["records"]
    profiles, filters, responses = {},{},{}
    for meta,self_mic,ref in timing.spoken.extra_cases():
        bot = meta["bot"]
        if meta["mode"]!="self_only" or bot in profiles:
            continue
        src = writer(ref,meta["cue_at"],0)
        coeff = timing.names.fit.fit(self_mic,src).astype("<f4")
        prediction = timing.names.estimate(src,coeff)
        test = slice(timing.names.fit.FIT_END,timing.names.fit.READY)
        val = timing.spoken.explained(self_mic[test].astype(float),prediction[test].astype(float))
        assert val["rms"]>=64 and val["explained"]>=.8
        profiles[bot] = dict(validation=val,coeff_sha256=ch.digest(coeff),mic_sha256=ch.digest(self_mic),
                             write_sha256=ch.digest(src),calibration_samples=timing.names.fit.FIT_END)
        filters[bot]=coeff
        room,name,rir = ch.read_room(Path(meta["room"]).name[:5])
        assert name==meta["room"] and rir==meta["rir_sha256"]
        responses[bot] = (room,meta["mic_echo_gain"])
    rows,raw_cache,cache,reads=[],{},{},{}
    totals=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    with np.load(timing.names.COEFF,allow_pickle=False) as calibrated:
        for row,mic,ref,_ in inputs:
            bot,at = row["bot"],row["start"]
            adc = prior.electrical(ref,at,"reference_lost")
            physical = timing.names.estimate(adc,calibrated[bot])
            h,gain=responses[bot]
            muted=ref.copy()
            muted[at:]=0
            tail=fftconvolve(muted.astype(float),h)[:len(ref)]*gain
            if row["role"]!="self":
                s = plan[row["index"]]
                assert s["target_id"]==row["role"]
                wav=timing.spoken.natural.replay.probe.dialogue.piper.load_wav(s)
                clip=wav[s["start"]:]
                length=s["end"]-s["start"]
                echo=fftconvolve(ref.astype(float),h)[:len(ref)]*gain
                factor=10**(row["db"]/20)*np.linalg.norm(echo[at:at+length])/np.linalg.norm(clip[:length].astype(float))
                external=np.zeros(len(ref))
                external[at:at+len(clip)] = factor*clip
                np.testing.assert_array_equal(np.rint(external+echo).clip(-32768,32767).astype("<i2"),mic)
                tail+=external
            muted_mic=np.rint(tail).clip(-32768,32767).astype("<i2")
            for mode,extra,actual in (("stable_writer",0,mic),("writer_20ms_earlier",320,mic),
                                      ("writer_20ms_later",-320,mic),("speaker_muted",0,muted_mic)):
                sha=ch.digest(actual)
                digital=writer(ref,at,extra)
                prediction=timing.names.estimate(digital,filters[bot])
                signals,choices=fallback(actual,adc,physical,prediction,parent["cutoff_samples"])
                if sha not in raw_cache:
                    raw_cache[sha]=net.score(actual)["c161"]
                result=dict(mode=mode,role=row["role"],voice=row["voice"],db=row["db"],command=row["command"],start=at,
                            command_at=row["command_at"],mic_sha256=sha,adc_sha256=ch.digest(adc),
                            write_sha256=ch.digest(digital),chosen_block_starts=choices,raw_c161=raw_cache[sha],routes={})
                for name,pcm in signals.items():
                    digest=ch.digest(pcm)
                    key=(sha,digest)
                    if key not in cache:
                        cache[key]=score(net,raw_cache[sha],pcm,row,spec,parent)
                    events=cache[key]
                    hits=[t for t in events if at<=t<row["end"]]
                    if name=="electrical" and mode!="speaker_muted":
                        old_run=lost[sha]["routes"]["static"]
                        assert old_run["residual_sha256"]==digest and old_run["events"]==events
                    record=dict(pcm_sha256=digest,events=events,hits=hits)
                    timely=bool(hits and row["command_at"] and hits[0]<row["command_at"])
                    total=totals[row["role"]][mode][name]
                    total.update(trials=1,hits=bool(hits),timely=timely)
                    if row["role"]=="Hey Nyx" and hits:
                        first=max(0,((hits[0]-HOP+159)//160)*160-32000)
                        cap=pcm[first:row["end"]]
                        ck=ch.digest(cap)
                        if ck not in reads:
                            reads[ck]=timing.names.transcripts(reader,cap)
                        words=timing.names.normalized(reads[ck]["text"])
                        want=timing.names.normalized(row["command"])
                        record.update(first=first,capture_sha256=ck,complete=words[-len(want):]==want)
                        total.update(complete=record["complete"],timely_complete=timely and record["complete"])
                    result["routes"][name]=record
                rows.append(result)
            print(len(rows),row["role"],row["voice"],flush=True)
    assert len(rows)==104 and len(profiles)==2
    with COEFF.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out,**filters)
    COEFF.with_suffix(".tmp").replace(COEFF)
    ch.save(OUT,dict(records=rows,summary=totals,texts=reads,profiles=profiles,asr=asr,models=net.identity,
        coeff_sha256=ch.sha(COEFF),sources={p.name:ch.sha(p) for p in (prior.OUT,timing.names.OUT,prior.fitted.COEFF,Path(__file__))},
        policy="Simulated post-gain local speaker writes arrive 40 ms ahead of playout throughout panel-only calibration. "
        "Separate causal FIR fitted on 4 s and validated on 750 ms. After electrical reference loss, either keep its filter, "
        "subtract the writer prediction when two ADC blocks are all zero, or also require a measured reduction of mic "
        "residual energy over the latest eight blocks. Include +/-20 ms write-clock jumps and writes that continue while "
        "the actual speaker is muted. All routes have the prior common 16 ms output delay, C161 and frozen linear early gate. "
        "Host PCM suffix ASR. These write-clock leads, loss signatures and room paths are simulated."))


if __name__=="__main__":
    main()
