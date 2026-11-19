"""Select one literal synthetic Nyx per voice on separate training rooms before paired room evaluation."""
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import json
import re
from zipfile import ZIP_DEFLATED, ZipFile

from kokoro_onnx import Kokoro
import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

from check_c161_command_playback_level import levels
from check_c161_wake_phone_duration_v1 import session
from coded_close_join import active
import diagnose_lewis_room_misses as room
from package_context_playback import add, formatted

ch, rooms, gap = room.ch, room.rooms, room.gap
STEM = "4090-c161-room-selected-spoken"
PACK = ch.RESULTS / (STEM + "-pool.zip")
FIT = ch.RESULTS / (STEM + "-fit.json")
MODEL = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
STYLES = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
VOICES = ("af_heart", "am_onyx", "bf_isabella", "bm_lewis", "af_river", "am_eric", "bf_lily", "bm_fable")
CONTEXTS = ("please begin", "are you ready")
SPEEDS = (1.0, .85, 1.15)
NAMES = ("Hey Nyx", "Hey Nick")


def build_pool():
    identity = ch.read(ch.RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    assert all(ch.sha(p) == identity["files"][str(p)] for p in (MODEL, STYLES))
    engine = session(str(MODEL))
    synth = Kokoro.from_session(engine, str(STYLES))
    candidates = []
    for voice in VOICES:
        for context in CONTEXTS:
            for speed in SPEEDS:
                for name in NAMES:
                    text = f"{name} {context}"
                    phones = synth.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
                    ids = synth.tokenizer.tokenize(phones)
                    assert len(phones) == len(ids)
                    audio, durations = engine.run(None, dict(input_ids=np.array([[0,*ids,0]], np.int64),
                        style=np.asarray(synth.get_voice_style(voice)[len(ids)], np.float32), speed=np.array([speed], np.float32)))
                    pcm = room.quantize(resample_poly(audio.ravel(), 2, 3)*32768)
                    boundary = list(re.finditer(r"\S+", phones))[2].start()
                    while phones[boundary] in "ˌˈ":
                        boundary += 1
                    edge = int(np.cumsum(durations)[boundary])*400
                    at, end = active(pcm[:edge])
                    clip = pcm[at:end]
                    meta = dict(index=len(candidates), voice=voice, context=context, speed=speed, name=name,
                                phones=phones, durations=durations.ravel().tolist(), source_pcm_sha256=ch.digest(pcm),
                                active_start=at, active_end=end, source_levels=levels(clip))
                    candidates.append((meta, clip))
    assert len(candidates) == 96
    # Common active level with 1 dB peak headroom for every target and control.
    level = min(-18.58, *(r["source_levels"]["active_rms_dbfs"] - 1 - r["source_levels"]["peak_dbfs"]
                           for r, _ in candidates))
    entries = []
    with ZipFile(PACK.with_suffix(".tmp"), "w", ZIP_DEFLATED) as z:
        for r, clip in candidates:
            scale = 10**((level - r["source_levels"]["active_rms_dbfs"])/20)
            q = np.rint(clip*scale)
            assert abs(q).max() <= 32767
            cue = np.pad(q.astype("<i2"), (0,gap.GAP))
            b = BytesIO()
            sf.write(b, cue, 16000, format="WAV", subtype="PCM_16")
            dest = f'{r["index"]:02}.wav'
            add(z, dest, b.getvalue())
            entries.append(dict(**r, gain=scale, file=dest, pcm_sha256=ch.digest(cue), samples=len(cue)))
        add(z, "manifest.json", formatted(dict(records=entries, active_rms_dbfs=level, gap=gap.GAP,
            sources={p.name:ch.sha(p) for p in (MODEL, STYLES, Path(__file__))}, provider=engine.get_providers()[0])))
    PACK.with_suffix(".tmp").replace(PACK)


def pool():
    with ZipFile(PACK) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read("manifest.json"))
        for r in manifest["records"]:
            cue, rate = sf.read(BytesIO(z.read(r["file"])), dtype="int16")
            assert rate == 16000 and ch.digest(cue) == r["pcm_sha256"] and not cue[-gap.GAP:].any()
            yield r, cue


def noise_training(basename):
    ledger = ch.read(ch.RESULTS/"4090-backgrounds-manifest.json")
    record, = [r for r in ledger["records"] if r["split"] == "training" and Path(r["path"]).name == basename]
    p = Path(".local/wakeword/backgrounds") / record["path"]
    assert ch.sha(p) == record["sha256"]
    data, rate = sf.read(p, always_2d=True)
    ratio = Fraction(16000, rate)
    return resample_poly(data.mean(1), ratio.numerator, ratio.denominator), record["sha256"]


def mixtures():
    length = 64000
    t = np.arange(length)/16000
    # Deterministic sustained major triads and 4 Hz amplitude modulation for a tonal masker.
    fundamental = np.where(t < 1.5, 130.81, 174.61)
    phase = 2*np.pi*np.cumsum(fundamental)/16000
    tonal = sum(np.sin(phase*interval*k)/k for interval in (1., 1.25, 1.5) for k in range(1,9))
    tonal *= .7 + .3*np.cos(2*np.pi*4*t)
    air = noise_training("AirConditioner_4.wav")
    vacuum = noise_training("VacuumCleaner_3.wav")
    noises = [("ac4", *air), ("vacuum3", *vacuum), ("triads", tonal, ch.digest(tonal))]
    for i, rir in enumerate(("h001_", "h007_", "h025_")):
        h, path, sha = rooms.read_room(rir)
        for j, (name, bg, noise_hash) in enumerate(noises):
            cut = (3721*i + 1173*j + np.arange(length)) % len(bg)
            v = bg[cut]
            v = v / np.sqrt(np.mean(v*v))
            for snr in (3, 9):
                # A constant reference power across voices and settings, at two playback phases.
                end = 48000 + 73*i + 181*j + (211 if snr == 9 else 0)
                noise = v * (32768*10**((-18.58-snr)/20))
                yield dict(room=path, rir_sha256=sha, noise=name, noise_sha256=noise_hash,
                           nominal_snr_db=snr, end=end), h, noise
    yield dict(room="dry", noise="quiet", end=48000), np.array([1.]), np.zeros(length)


def main():
    if not PACK.exists():
        build_pool()
    native = rooms.c161_binding()
    recipes = list(mixtures())
    rows, choices = [], {}
    for r, cue in pool():
        tests = []
        for condition, h, bg in recipes:
            end = condition["end"]
            start = end-len(cue)
            mic = bg.copy()
            wet = fftconvolve(cue.astype(float), h)
            n = min(len(wet), len(mic)-start)
            mic[start:start+n] += wet[:n]
            pcm = room.quantize(mic)
            test = room.measure(native, pcm, start, end)
            tests.append(dict(timely=test["timely"], events=test["events"], peak5=test["peak5"],
                              pcm_sha256=ch.digest(pcm), clipped=int((np.abs(mic)>32767).sum())))
        rows.append(dict(index=r["index"], voice=r["voice"], speed=r["speed"], context=r["context"],
                         name=r["name"], tests=tests))
        if len(rows) % 12 == 0:
            group = rows[-12:]
            rankings = []
            for pos, neg in zip(group[::2], group[1::2], strict=True):
                target = pos["tests"]
                control = neg["tests"]
                # Rank by paired correct behavior first, then ordinary target support.
                key = [sum(p["timely"] and not n["events"] for p,n in zip(target,control,strict=True)),
                       sum(p["timely"] for p in target), -sum(bool(n["events"]) for n in control),
                       int(np.median([p["peak5"] for p in target])), -pos["index"]]
                rankings.append(dict(target=pos["index"], nick=neg["index"], rank=key))
            best = max(rankings, key=lambda x:x["rank"])
            choices[r["voice"]] = dict(selected=best, default=rankings[0], rankings=rankings)
            print(r["voice"], best, flush=True)
    assert len(choices) == 8
    ch.save(FIT, dict(records=rows, choices=choices, conditions=[s for s,_,_ in recipes],
                      pool_sha256=ch.sha(PACK), model_sha256=ch.sha(rooms.MODEL), script_sha256=ch.sha(Path(__file__)),
                      policy="Selection uses three training RIRs distinct from both downstream rooms, two training "
                      "noise files and analytic triads. Eight exposed synthetic voice identities; six pronunciation "
                      "settings scored per voice. Pick by count of paired timely target and no Nick event, then "
                      "target hits, fewer Nick events, median target peak sum, then earliest source index. "
                      "Freeze indices before replay on downstream room scenes. No model update or holdout access."))


if __name__ == "__main__":
    main()
