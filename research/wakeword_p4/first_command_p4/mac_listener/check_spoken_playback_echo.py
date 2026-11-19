"""Check whether a fitted playback echo explains a wake on ordinary spoken Bot audio."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, correlate, fftconvolve, sosfilt

import connected_nick_willow as connected
import diagnose_early_nick as natural
import coded_fixed_advance as fixed
import check_c161_lexical_echo as old_echo
from check_c161_bot_replies_v1 import STEM as BOT
from replay_c161_pcm_cues_v1 import detect

ch = fixed.ch
OUT = ch.RESULTS/"4090-c161-spoken-bot-echo.json"
LOWPASS = butter(5, 1500, fs=16000, output="sos")
TAPS = 512  # 128 ms on the 4 kHz analysis clock.


def down(pcm):
    return sosfilt(LOWPASS, pcm.astype(float))[3::4]


def explained(actual, prediction):
    power = float(actual @ actual)
    residual = actual-prediction
    fraction = 1-float(residual @ residual)/max(1., power)
    return dict(rms=float(np.sqrt(power/len(actual))), explained=fraction)


def echo_check(mic, ref, clocks):
    # Both causal filters and the fitted impulse use only samples before each decision.
    m, r = down(mic), down(ref)
    n = 4*4000
    x, y = r[:n], m[:n]
    xx = correlate(x,x,mode="full",method="fft")[n-1:n-1+TAPS].copy()
    yx = correlate(y,x,mode="full",method="fft")[n-1:n-1+TAPS]
    xx[0] += max(1., .001*xx[0])
    h = solve_toeplitz(xx, yx)
    predicted = fftconvolve(r,h)[:len(m)]
    # A separate pre-trigger window tests whether the path learned from the first
    # four seconds of panel audio predicts the following 750 ms.
    valid = explained(m[n:19000],predicted[n:19000])
    calibrated = valid["rms"] >= 64 and valid["explained"] >= .80
    windows = {}
    for t in clocks:
        stop = t//4
        parts = []
        if stop >= 19000+2*640 and calibrated:
            for end in (stop-640,stop):
                parts.append(explained(m[end-640:end],predicted[end-640:end]))
        veto = len(parts)==2 and all(v["rms"]>=64 and v["explained"]>=.90 for v in parts)
        windows[t] = dict(veto=veto,parts=parts)
    return dict(calibrated=calibrated, calibration=valid, events=windows)


def daily_and_bot(net, coded):
    study = natural.study
    native_history = ch.read(ch.RESULTS/(study.training.STEM+"-replay.json"))["records"]
    bot_pack = ch.RESULTS/(BOT+"-pcm-v1.npz")
    assert ch.sha(bot_pack)==ch.read(ch.RESULTS/(BOT+"-v1.json"))["pcm_archive_sha256"]
    rows = []
    with np.load(bot_pack) as bank:
        for group in ("daily","bot"):
            for (meta,pcm,c161,_,_),old in zip(natural.base.sources(group), native_history[group], strict=True):
                src=meta["source"]
                score=net.score(pcm)
                assert ch.digest(pcm)==old["pcm_sha256"] and score["c161"]==c161 and score["events"]==old["events"]["calibrated"]
                code=[e["t"] for e in detect(coded,[pcm])]
                veto={}
                if group=="bot":
                    dry=bank[src["key"].rsplit("_",1)[0]+"_digital"]
                    veto=echo_check(pcm,dry,sorted(set(score["events"]+code)))
                parts=[]
                for s in src["spans"]:
                    def within(events):
                        return [t for t in events if s["start"]<=t<s["end"]+6400]
                    events=within(score["events"])
                    parts.append(dict(text=s["text"],role=s["address"],c161=within(c161),v5=within(code),
                                      natural=events,after_echo=[t for t in events if not veto or not veto["events"][t]["veto"]]))
                rows.append(dict(group=group,index=old["index"],mic_sha256=ch.digest(pcm),source=src,
                                 c161=c161,v5=code,natural=score["events"],echo=veto,spans=parts))
    return rows


def extra_cases():
    # Overlay each four-voice natural/fixed pair onto a Bot deliberately mentioning the wake name.
    # Both source and self-only controls share a five-second panel calibration before the overlap.
    plan=ch.read(connected.PLAN)
    input_report=ch.read(ch.RESULTS/(connected.STEM+".json"))
    assert ch.sha(connected.PLAN)==input_report["plan_sha256"]
    bank=ch.RESULTS/(BOT+"-pcm-v1.npz")
    bot_history=ch.read(ch.RESULTS/(BOT+"-v1.json"))
    assert ch.sha(bank)==bot_history["pcm_archive_sha256"]
    with np.load(bank) as tracks:
        for j,voice in enumerate(connected.VOICES):
            key=("af_kore_digital","bm_fable_digital")[j%2]
            target, = [b for b in bot_history["records"] if b["key"]==key]
            dry=tracks[key]
            assert ch.digest(dry)==target["pcm_sha256"]
            s=target["spans"][-1]
            h,room,rir_sha=ch.read_room(("h012_","h004_")[j%2])
            at=5*16000
            n=10*16000
            ref=np.zeros(n,dtype="<i2")
            ref[:at]=dry[:at]
            ref[at:at+s["end"]-s["start"]]=dry[s["start"]:s["end"]]
            echo=fftconvolve(ref.astype(float),h)[:n]
            shared_gain=min(1., 10000./max(1.,np.max(np.abs(echo))))
            echo*=shared_gain
            silence=np.zeros_like(ref)
            common=dict(voice=voice,bot=key,room=room,rir_sha256=rir_sha,mic_echo_gain=shared_gain,
                        bot_text=s["text"],reference_sha256=ch.digest(ref),cue_at=at,reference_start=at)
            yield dict(common,mode="self_only"),np.rint(echo).astype("<i2"),ref
            for role in ("Hey Nyx","fixed-v5"):
                src,=[s for s in plan["records"] if s["voice"]==voice and s["target_id"]==role
                      and s["carrier"]==connected.CARRIERS[0]]
                signal=natural.replay.probe.dialogue.piper.load_wav(src)
                clip=signal[src["start"]:]
                length=src["end"]-src["start"]
                assert src["pcm_sha256"]==ch.digest(signal) and len(clip)+at<=n
                for db in (-6,0,6):
                    gain=10**(db/20)*np.linalg.norm(echo[at:at+length])/np.linalg.norm(clip[:length].astype(float))
                    external=np.zeros(n)
                    external[at:at+len(clip)]=gain*clip
                    for mode,mix,digital in (("external_only",external,silence),("overlap",external+echo,ref)):
                        q=np.rint(mix)
                        mic=q.clip(-32768,32767).astype("<i2")
                        info=dict(common,mode=mode,role=role,original_pcm_sha256=src["pcm_sha256"],
                                  command_at=at+length,carrier=src["carrier"],external_to_echo_db=db,
                                  external_gain=gain,clips=int(np.count_nonzero(q!=mic)))
                        yield info,mic,digital


def replay(net, coded, group, meta, mic, ref):
    score=net.score(mic)
    fixed_events=[r["t"] for r in detect(coded,[mic])]
    panel=natural.base.scoring.merged(score["c161"],fixed_events)
    clocks=sorted(set(panel+score["events"]))
    guard=echo_check(mic,ref,clocks)
    routes={"panel":panel, "natural":score["events"]}
    return dict(group=group,source=meta,mic_sha256=ch.digest(mic),ref_sha256=ch.digest(ref),c161=score["c161"],
                v5=fixed_events, original=routes, echo=guard,
                kept={k:[t for t in ev if not guard["events"][t]["veto"]] for k,ev in routes.items()})


def main():
    net=natural.Detector()
    assert net.identity["c161"]==natural.MODEL_SHA
    with TemporaryDirectory() as tmp:
        coded=ch.binding(Path(tmp)/"v5.so",coded="v5")
        known=daily_and_bot(net,coded)
        pairs=[]
        for key,mic,ref,meta in old_echo.cases():
            assert (ch.digest(mic),ch.digest(ref))==(meta["mic_sha256"],meta["ref_sha256"])
            pairs.append(replay(net,coded,"archived_pairs",dict(meta,key=key),mic,ref))
        for meta,mic,ref in extra_cases():
            pairs.append(replay(net,coded,"additional_pairs",meta,mic,ref))
    assert len(known)==100 and len(pairs)==76
    ch.save(OUT,dict(known=known, pairs=pairs, models=net.identity, script_sha256=ch.sha(Path(__file__)),
        prior_sha256=ch.sha(ch.RESULTS/(natural.study.training.STEM+"-replay.json")), connected_plan_sha256=ch.sha(connected.PLAN),
        bot_sha256=ch.sha(ch.RESULTS/(BOT+"-pcm-v1.npz")), policy="Diagnostic host echo attribution. Fit a 128 ms "
        "causal linear filter on the first 4 seconds at 4 kHz; validate on the next 750 ms. At a wake clock require "
        "at least 90% variance explained in each of the last two 160 ms windows, with RMS at least 64. Validation "
        "requires 80% and RMS 64. No new delay. A static path, known panel-only calibration and aligned reference "
        "are assumptions. Ordinary voices, gains and controls are reused. Apply only at actual detector clocks. "
        "A notification during simultaneous mentions can belong to either source."))


if __name__=="__main__":
    main()
