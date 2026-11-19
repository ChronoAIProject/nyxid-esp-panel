"""Freeze paired isolated and sentence-conditioned address clips ahead of a common command."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
STEM = "4090-c161-synthesized-call-context"
PLAN = ROOT/"results"/(STEM+"-plan.json")
WORK = Path(".local/wakeword/c161-synthesized-call-context")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())
VOICES = ("af_heart","am_onyx","bf_isabella","bm_lewis")
NAMES = ("Hey Nyx","Hi Nicks","Hey Nick","Hey Willow")
COMMANDS = ("can you unlock the side door?", "could you check the time for me?", "turn down the hall light.")


def prepare():
    import re
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly
    from check_c161_wake_phone_duration_v1 import session
    from check_c161_command_playback_level import levels
    from coded_close_join import active
    import check_early_gate_voices as voices

    ch = voices.ch
    WORK.mkdir(exist_ok=True)
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    inventory = ch.read(ch.RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    assert all(inventory["files"][str(p)]==ch.sha(p) for p in (model,styles))
    engine = session(str(model))
    synth = Kokoro.from_session(engine,str(styles))
    original = ch.read(voices.study.matched.PLAN)
    files = {p:original["files"][p] for p in original["runtime"].values()}
    for p in (RUNNER,model,styles,voices.gate.fit.COEFF,voices.study.matched.PLAN):
        files[str(p)] = ch.sha(p)

    def render(text,voice):
        phones = synth.tokenizer.phonemize(text,"en-gb" if voice.startswith("b") else "en-us")
        ids = synth.tokenizer.tokenize(phones)
        assert len(phones)==len(ids)
        samples,durations = engine.run(None,dict(input_ids=np.array([[0,*ids,0]],np.int64),
            style=np.asarray(synth.get_voice_style(voice)[len(ids)],np.float32),speed=np.array([1.],np.float32)))
        wave = np.rint(resample_poly(samples.ravel(),2,3)*32768).clip(-32768,32767).astype("<i2")
        assert len(wave)==int(durations.sum())*400
        return wave,dict(text=text,phones=phones,durations=durations.ravel().tolist(),pcm_sha256=ch.digest(wave))

    info, clips, tails = {},{},{}
    for voice in VOICES:
        for name in NAMES:
            for command in ("",*COMMANDS):
                text = name+" "+command if command else name+"."
                wave, meta = render(text,voice)
                if command:
                    words = list(re.finditer(r"[^ ,.!?]+",meta["phones"]))
                    at = words[2].start()
                    while meta["phones"][at] in "ˈˌ":
                        at += 1
                    edge = int(np.cumsum(meta["durations"])[at])*400
                    if name=="Hey Nyx":
                        tails[voice,command] = wave[edge:]
                    meta["command_at"] = edge
                    prefix = wave[:edge]
                else:
                    prefix = wave
                lo,hi = active(prefix)
                clip = prefix[lo:hi]
                key = (voice,name,command)
                clips[key] = clip
                info[key] = dict(meta,active=[lo,hi],pre_gain=levels(clip),original_active_sha256=ch.digest(clip))
    db = min(original["matched_active_rms_dbfs"],*(m["pre_gain"]["active_rms_dbfs"]-1-m["pre_gain"]["peak_dbfs"] for m in info.values()))
    command_at = 48000
    rows = []
    for voice in VOICES:
        for command in COMMANDS:
            tail = np.pad(tails[voice,command],(0,16000))
            configs = [(name,style) for name in NAMES for style in ("isolated","connected")] + [("silence","control")]
            for name,style in configs:
                mic = np.r_[np.zeros(command_at,np.int16),tail]
                start = command_at
                cue_info = {}
                if name!="silence":
                    key = (voice,name,command if style=="connected" else "")
                    cue = clips[key]
                    gain = 10**((db-info[key]["pre_gain"]["active_rms_dbfs"])/20)
                    q = np.rint(cue*gain)
                    assert abs(q).max() <= 32767
                    start -= len(q)
                    assert start >= 16000
                    mic[start:command_at] = q.astype("<i2")
                    cue_info = dict(**info[key], gain=gain, cue_sha256=ch.digest(mic[start:command_at]))
                i = len(rows)
                path = WORK/f"{i:03}.wav"
                sf.write(path,mic,16000,subtype="PCM_16")
                files[str(path)] = ch.sha(path)
                rows.append(dict(index=i,voice=voice,target_id=name,style=style,carrier=command,start=start,end=command_at,
                                 path=str(path),sha256=files[str(path)],pcm_sha256=ch.digest(mic),samples=len(mic),
                                 suffix_sha256=ch.digest(tail),cue=cue_info))
    assert len(rows)==108
    result = dict(files=files,records=rows,runtime=original["runtime"],threshold_requests=["DEFAULT"],
        matched_active_rms_dbfs=db,provider=engine.get_providers()[0],speed=1.,
        policy="Four already exposed development voices. Render each name alone or in the same unpunctuated "
        "sentence as the following command. Cut before the duration-model command token, trim both cue types "
        "at the same -30 dB relative frame power and normalize active RMS. Every voice/command cell shares "
        "the exact literal-Nyx command suffix and onset. Add a silence control. Compare frozen gates, raw C161 "
        "and official Willow with no changes to weights, thresholds or rates. No untouched holdout.")
    if PLAN.exists():
        assert ch.read(PLAN)==result
    else:
        ch.save(PLAN,result)


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="prepare":
        prepare()
    elif action=="case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN,str(RUNNER)
        official.case()
    elif action=="willow":
        import run_c161_willow_matched_scenes_v1 as official
        official.PLAN, official.OUT, official.__file__ = PLAN,WORK/"runs",str(RUNNER)
        official.willow()
    else:
        raise SystemExit("prepare | willow | case")
