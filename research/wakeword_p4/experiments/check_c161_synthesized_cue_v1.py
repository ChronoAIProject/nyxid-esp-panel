"""Qualify one selected original synthetic WAV under fixed host playback conditions."""
import hashlib
import json
import wave

import numpy as np
import soundfile as sf

from generate_c161_vowel_name_pair_v1 import AUDIO, RESULTS, SOURCE, sha
from replay_c161_vowel_name_pair_v1 import binding
import verify_c161_ambient_stream_v1 as live
from review_c161_equal_length_history_v1 import MODEL, MODEL_SHA
from audit_c161_history_paths_v1 import decode, arbitrate, times


def main():
    assert sha(MODEL) == MODEL_SHA
    r = json.loads(SOURCE.read_bytes())["records"][0]
    assert (r["kind"], r["text"], r["voice"], r["speed"]) == ("nyx", "Hey Nicks.", "af_bella", 0.9)
    with np.load(AUDIO) as z:
        cue = z[r["key"]].copy()
    assert hashlib.sha256(cue.tobytes()).hexdigest() == r["pcm_sha256"]
    path = RESULTS / "hye-nyx-c161-synthetic-cue-v1.wav"
    with path.open("xb") as handle, wave.open(handle, "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(cue.tobytes())
    decoded, rate = sf.read(path, dtype="int16")
    np.testing.assert_array_equal(decoded, cue)
    assert rate == 16000
    native = binding()
    admission = json.loads((RESULTS / "4090-chatterbox-daily-training-admission-v1.json").read_bytes())
    prefixes = [("silence", np.zeros(32000, np.int16), [])]
    for j, voice in enumerate(("af_bella","am_onyx","bf_isabella","bm_fable")):
        choices = [row for row in admission["admitted"] if row["request"]["voice"] == voice
                   and row["request"]["group"] == "conversation"]
        row = choices[j]
        from pathlib import Path
        ab = Path(row["wav"]["path"])
        assert sha(ab) == row["wav"]["sha256"]
        prefix, rate = sf.read(ab, dtype="int16")
        assert rate == 16000
        prefixes.append((voice, prefix, [row["request"]["text"],sha(ab)]))
    records, arrays = [], {}
    for voice, prefix, provenance in prefixes:
        for gap in (4000, 8000):
            for gain_db in (0, -6, -12):
                target = cue.astype(float) * 10**(gain_db/20)
                start = len(prefix) + gap
                dry = np.r_[prefix, np.zeros(gap), target, np.zeros(8000)]
                for noise in (False, True):
                    # The same underlying noise draw is used across playback levels.
                    raw = dry.copy()
                    if noise:
                        g = np.random.default_rng(20261118 + start)
                        raw += g.normal(size=len(raw)) * np.sqrt(np.mean(cue.astype(float)**2)) / 10
                    quant = np.rint(raw)
                    pcm = np.clip(quant, -32768, 32767).astype("<i2")
                    codes, clock, emitted, suppressed = live.run_stream(native, pcm)
                    c, a = np.array(codes, np.uint8)
                    clock = np.asarray(clock)
                    base, alt = decode(c, clock), decode(a, clock)
                    assert arbitrate(base, alt) == (emitted, suppressed)
                    i = str(len(records))
                    arrays[i+"_scores"] = np.stack((c, a))
                    arrays[i+"_clock"] = clock
                    records.append(dict(key=i, history=voice, source=provenance, gap=gap, gain_db=gain_db,
                                        fixed_noise=noise, clipped=int(((quant > 32767) | (quant < -32768)).sum()),
                                        pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                        cue_start=start, cue_end=start+len(cue), original=times(base),
                                        ambient=times(alt), pair=times(emitted)))
    scores = RESULTS / "4090-c161-synthetic-cue-scores-v1.npz"
    with scores.open("xb") as f:
        np.savez_compressed(f, **arrays)
    report = dict(model_sha256=MODEL_SHA, wav_sha256=sha(path), pcm_sha256=r["pcm_sha256"], samples=len(cue),
                  generation_source_sha256=sha(SOURCE), source_key=r["key"], scores_sha256=sha(scores), records=records)
    with (RESULTS/"4090-c161-synthetic-cue-v1.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
