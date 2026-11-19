"""Render matched synchronous mic and speaker-reference samples after fixing the capture rule."""
import io
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import replay as listener

RATE = 16000
STEM = "4090-c161-mac-playback-transition"
WORK = listener.WORK / "transition"
PLAN = STEM + "-inputs.json"
AUDIO = STEM + "-stereo.zip"
COMMANDS = (
    ("af_heart", "Move the reading light closer to the window."),
    ("bm_lewis", "Set the hallway lights to twenty percent."),
)
ROOMS = (("h012_", "h018_", "air_conditioner"), ("h004_", "h010_", "music"))
PANEL = "Your afternoon delivery is scheduled for today. The front desk will send a message when the package arrives."
SCHEDULES = ("quiet", "start", "stop", "burst")
OLD = listener.RESULTS / "4090-c161-v5-playback-reference.json"


def commands():
    from kokoro_onnx import Kokoro
    from check_c161_wake_phone_duration_v1 import session

    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    inventory = listener.read(listener.RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    for p in (model, styles):
        assert listener.sha(p) == inventory["files"][str(p)]
    training = {r["request"]["conditioning_id"] for r in inventory["requests"] if r["request"]["split"] == "training"}
    engine = session(str(model))
    synth = Kokoro.from_session(engine, str(styles))
    rendered = []
    for voice, text in (*COMMANDS, ("af_kore", PANEL)):
        assert voice in training
        phones = synth.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
        ids = synth.tokenizer.tokenize(phones)
        wave, durations = engine.run(None, dict(input_ids=np.array([[0, *ids, 0]], np.int64),
            style=np.asarray(synth.get_voice_style(voice)[len(ids)], np.float32), speed=np.array([1.], np.float32)))
        mono = resample_poly(wave.ravel(), 2, 3)
        assert len(mono) == int(durations.sum()) * 400 and np.isfinite(mono).all()
        rendered.append(np.rint(mono * 32768).clip(-32768, 32767).astype("<i2"))
    return rendered, {str(p): listener.sha(p) for p in (model, styles)}


def read_room(prefix):
    r, = [r for r in listener.read(listener.RESULTS / "4090-mit-rir-audit.json")["records"]
          if r["split"] == "training" and Path(r["path"]).name.startswith(prefix)]
    path = Path(".local/wakeword/mit-rirs") / r["path"]
    assert listener.sha(path) == r["sha256"]
    h, rate = sf.read(path)
    assert rate == RATE and h.ndim == 1
    return h / np.linalg.norm(h), r["path"], r["sha256"]


def read_noise(kind):
    if kind == "music":
        record = listener.read(listener.RESULTS / "4090-c161-music-development-sources-v1.json")["records"][0]
        path = Path(record["path"])
    else:
        record, = [r for r in listener.read(listener.RESULTS / "4090-backgrounds-manifest.json")["records"]
                   if r["path"] == "noise_train/AirConditioner_2.wav" and r["split"] == "training"]
        path = Path(".local/wakeword/backgrounds") / record["path"]
    assert listener.sha(path) == record["sha256"]
    x, rate = sf.read(path, always_2d=True)
    ratio = Fraction(RATE, rate)
    return resample_poly(x.mean(axis=1), ratio.numerator, ratio.denominator), str(path), record["sha256"]


def challenge(waves, files):
    cue, rate = sf.read(listener.CUE, dtype="int16")
    assert rate == RATE and listener.sha(listener.CUE) == "79c37ecc57e3a7297f1a9ff1a9481fd56af1b985872a7875ccbadb5ffbf40f47"
    play = waves[-1].astype(float)
    start = 2 * RATE
    onset = start + len(cue) + 320
    for dry_rir, echo_rir, noise_kind in ROOMS:
        h, room, files[dry_rir] = read_room(dry_rir)
        g, echo_room, files[echo_rir] = read_room(echo_rir)
        noise, noise_path, noise_hash = read_noise(noise_kind)
        files[noise_path] = noise_hash
        for vi, (voice, carrier) in enumerate(COMMANDS):
            dry = waves[vi]
            command = fftconvolve(dry.astype(float), h)
            length = onset + len(command) + RATE
            external = np.zeros(length)
            wet_cue = fftconvolve(cue.astype(float), h)
            external[start:start+len(wet_cue)] = wet_cue
            external[onset:onset+len(command)] += command
            background = noise[(6+vi*4)*RATE:][:length]
            assert len(background) == length
            powers = np.mean(np.pad(command**2, (0, -len(command)%320)).reshape(-1, 320), axis=1)
            active = np.repeat(powers >= powers.max()*.001, 320)[:len(command)]
            def rms_ratio(track):
                return np.linalg.norm(command[active]) / np.linalg.norm(track[onset:onset+len(command)][active])
            noise_gain = rms_ratio(background) * 10**(-6/20)
            panel = np.resize(play, length)
            # One continuous underlying playback timeline and gain for all four schedules.
            echo_gain = rms_ratio(fftconvolve(panel, g)[:length])
            mid = onset + len(dry)//2
            burst = (onset + len(dry)//4, onset + 3*len(dry)//4)
            windows = ((0, 0), (mid, length), (0, mid), burst)
            tracks = []
            for lo, hi in windows:
                ref = np.zeros(length)
                ref[lo:hi] = panel[lo:hi]
                mic = external + noise_gain*background + echo_gain*fftconvolve(ref, g)[:length]
                tracks.append(np.column_stack((mic, ref)))
            gain = min(1., 28000/max(np.abs(v).max() for v in tracks))
            for timing, window, pair in zip(SCHEDULES, windows, tracks, strict=True):
                yield np.rint(gain*pair).astype("<i2"), dict(
                    group="challenge", timing=timing, role="external", voice=voice, carrier=carrier,
                    cue_at=start, command_at=onset, dry_command_end=onset+len(dry), panel_window=window,
                    room=room, echo_room=echo_room, noise=noise_kind, noise_start=(6+vi*4)*RATE,
                    noise_gain=noise_gain, echo_gain=echo_gain, common_gain=gain,
                    dry_sha256=listener.digest(dry), panel_dry_sha256=listener.digest(waves[-1]))


def archived_controls():
    previous = listener.read(OLD)
    source = listener.RESULTS / "4090-c161-v5-playback-reference-stereo.zip"
    assert listener.sha(source) == previous["stereo_archive_sha256"]
    with ZipFile(source) as z:
        for r in previous["records"]:
            if r["reference"] != "speaker":
                continue
            pair, rate = sf.read(io.BytesIO(z.read(r["key"]+".wav")), dtype="int16")
            assert rate == RATE and listener.digest(pair[:,0]) == r["mic_sha256"]
            assert listener.digest(pair[:,1]) == r["reference_sha256"]
            yield pair, dict(group="regression", previous_key=r["key"], role=r["role"], room=r["room"],
                             cue_at=r["start"], command_at=r["command_onset"], carrier="Lower the fan speed.",
                             original_raw_v5=r["input_events"])


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    waves, files = commands()
    rows = []
    with ZipFile(output/AUDIO, "x") as audio:
        from itertools import chain
        for i, (pair, meta) in enumerate(chain(archived_controls(), challenge(waves, files))):
            buffer = io.BytesIO()
            sf.write(buffer, pair, RATE, format="FLAC", subtype="PCM_16")
            saved, rate = sf.read(io.BytesIO(buffer.getvalue()), dtype="int16")
            assert rate == RATE and np.array_equal(saved, pair)
            audio.writestr(f"{i:03}.flac", buffer.getvalue())
            rows.append(dict(index=i, **meta, samples=len(pair), mic_sha256=listener.digest(pair[:,0]),
                             ref_sha256=listener.digest(pair[:,1])))
    assert len(rows) == 22
    listener.save(output/PLAN, dict(records=rows, files=files, generator_sha256=listener.sha(Path(__file__)),
         archive_sha256=listener.sha(output/AUDIO), previous_sha256=listener.sha(OLD),
         model_sha256=listener.C161, cue_sha256=listener.sha(listener.CUE), panel_text=PANEL,
         provider="CUDAExecutionProvider", nominal_environment_snr_db=6,
         echo_db_on_continuous_ungated_command=0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK/"inputs")
    prepare(parser.parse_args().out)
