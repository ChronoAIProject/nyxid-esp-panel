"""Continuous development playback of synthetic assistant replies and explicit wake mentions."""
import hashlib
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

from audit_c161_history_paths_v1 import decode, times
from check_c161_stream_stretch_v1 import run_quantized
from check_c161_wake_phone_duration_v1 import session
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, read, save, sha

STEM = "4090-c161-bot-replies"
REPLIES = (
    "Hello there. How can I help you today?",
    "Your next appointment starts in six minutes.",
    "I've set the kitchen timer for six minutes.",
    "I turned the study lamp off.",
    "The next train leaves from platform six.",
    "Nick sent you a message about dinner.",
    "Max asked me to remind you about the keys.",
    "Nicole will pick you up at the station.",
    "I've added milk and eggs to your shopping list.",
    "The forecast is clear skies this afternoon.",
    "Your music is paused. Shall I play the next track?",
    "I couldn't hear that. Could you repeat your request?",
)
MENTIONS = ("To wake me, say Hey Nicks.", "Hey Nicks, turn on the kitchen lights.")


def main():
    config = read(RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(path) == config["files"][str(path)] for path in (bank, model))
    runtime = session(str(model))
    engine = Kokoro.from_session(runtime, str(bank))
    mid, = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"]
    detectors = {"c161": MODEL, "midlevel": RESULTS / mid["path"]}
    assert sha(MODEL) == MODEL_SHA and sha(detectors["midlevel"]) == mid["sha256"]
    arms = (("digital", [1.]), ("reflections", [0.45, 0.13, -0.06]))
    records, waves = [], {}
    for voice in ("af_kore", "bm_fable"):
        lang = "en-gb" if voice.startswith("b") else "en-us"
        clips, spans = [], []
        offset = 0
        for i, text in enumerate((*REPLIES, *MENTIONS)):
            phones = engine.tokenizer.phonemize(text, lang)
            ids = engine.tokenizer.tokenize(phones)
            raw, duration = runtime.run(None, dict(input_ids=np.array([[0, *ids, 0]], np.int64),
                style=np.asarray(engine.get_voice_style(voice)[len(ids)], np.float32), speed=np.ones(1, np.float32)))
            signal = resample_poly(raw.ravel(), 2, 3)
            assert len(signal) == duration.sum() * 400 and np.isfinite(signal).all()
            clip = np.rint(np.clip(signal, -1, 32767/32768)*32768).astype("<i2")
            clips.extend((clip, np.zeros(8000, np.int16)))
            spans.append(dict(text=text, group="reply" if i < len(REPLIES) else "explicit_mention",
                              start=offset, end=offset+len(clip), source_clipped=int(np.count_nonzero((signal < -1) | (signal > 32767/32768)))))
            offset += len(clip) + 8000
        dry = np.concatenate(clips)
        for arm, gains in arms:
            # Fixed 15 and 45 ms delayed reflections, keeping the same timeline.
            pcm = dry.copy()
            if arm == "reflections":
                wet = dry.astype(np.float64) * gains[0]
                for lag, gain in zip((240, 720), gains[1:], strict=True):
                    wet[lag:] += gain * dry[:-lag]
                assert np.abs(wet).max() < 32768
                pcm = np.rint(wet).astype("<i2")
            key = voice + "_" + arm
            waves[key] = pcm
            features, clock = frontend(pcm)
            events = {}
            for detector, path in detectors.items():
                scale, zero = REF_KERNEL(model_path=str(path)).get_input_details()[0]["quantization"]
                q = np.clip(np.rint(features / scale) + zero, -128, 127).astype(np.int8)
                for stretch in ((False, True) if detector == "midlevel" else (False,)):
                    scores, _ = run_quantized(q, path, stretch)
                    name = "expanded" if stretch else detector
                    events[name] = times(decode(scores, clock))
            records.append(dict(key=key, voice=voice, transport=arm, spans=spans, events=events,
                                samples=len(pcm), pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    out = RESULTS / f"{STEM}-pcm-v1.npz"
    with out.open("xb") as handle:
        np.savez_compressed(handle, **waves)
    save(RESULTS / f"{STEM}-v1.json", dict(tts_sha256=sha(model), bank_sha256=sha(bank), pcm_archive_sha256=sha(out),
         detectors={k:sha(p) for k,p in detectors.items()}, sample_rate=16000, records=records))


if __name__ == "__main__":
    main()
