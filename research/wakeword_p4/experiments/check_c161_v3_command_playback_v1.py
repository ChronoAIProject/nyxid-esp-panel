"""Check immediate commands after the frozen v3 playback cue through the existing detector."""
import hashlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
import replay_c161_pcm_cues_v1 as matcher
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
from verify_c161_ambient_stream_v1 import run_stream

STEM = "4090-c161-v3-command-playback"


def main():
    plan = RESULTS / "4090-c161-signed-forward-plan-v1.json"
    source = read(plan)
    waves = RESULTS / "4090-c161-signed-forward-inputs-v1.zip"
    assert sha(waves) == source["archive_sha256"] and sha(MODEL) == MODEL_SHA
    cue_file = RESULTS / "hye-nyx-c161-synthetic-cue-v3.wav"
    cue_sha = read(RESULTS / "4090-c161-suffix-cue-v3-source.json")["wav_sha256"]
    cue = load_wav(dict(path=cue_file, sha256=cue_sha))
    native, rows = binding(), []
    with ZipFile(waves) as archive, tempfile.TemporaryDirectory() as work, \
         patch.object(matcher, "LIB", Path(work) / "confirmed.so"):
        detector = matcher.binding(True)
        for index, r in enumerate(source["records"]):
            if r["target_id"] != "nyx":
                continue
            original = archive.read(Path(r["path"]).name)
            assert hashlib.sha256(original).hexdigest() == r["sha256"]
            pcm, rate = sf.read(io.BytesIO(original), dtype="int16")
            assert rate == 16000 and hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
            start = r["start"]
            command = pcm[r["first_phone"]:]
            first = start + len(cue)
            for channel in ("digital", "room_noise"):
                ir = np.zeros(577)
                ir[[0, 173, 576]] = [.65, .24, -.13]
                noise = np.random.default_rng(85370 + index).normal(size=first + len(command))
                noise = lfilter([1.], [1., -.75], noise)
                noise *= np.sqrt(np.mean(cue.astype(float)**2)/np.mean(noise**2)) * .065
                for playing in (True, False):
                    inserted = cue if playing else np.zeros_like(cue)
                    samples = np.r_[pcm[:start], inserted, command]
                    if channel == "room_noise":
                        samples = lfilter([.25, .5, .25], [1.], lfilter(ir, [1.], samples)) + noise
                    clipped = int(np.count_nonzero((samples > 32767) | (samples < -32768)))
                    samples = np.rint(samples).clip(-32768, 32767).astype("<i2")
                    events = matcher.detect(detector, (samples[i:i+480] for i in range(0, len(samples), 480)))
                    codes, clock, _, _ = run_stream(native, samples)
                    original_events = times(decode(np.asarray(codes[0], np.uint8), np.asarray(clock)))
                    row = dict(voice=r["voice"], scene=r["scene"], carrier=r["carrier"], channel=channel, playing=playing,
                               source_pcm_sha256=r["pcm_sha256"], pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                               samples=len(samples), cue_start=start, command_start=first, clipped=clipped, confirmed=events,
                               original=original_events, merged=matcher.joined(original_events, events))
                    if index == 0 and playing:
                        witness = RESULTS / f"{STEM}-{channel}-v1.wav"
                        with witness.open("xb") as f:
                            sf.write(f, samples, 16000, format="WAV", subtype="PCM_16")
                        row.update(wav=witness.name, wav_sha256=sha(witness))
                    rows.append(row)
    assert len(rows) == 48
    save(RESULTS / f"{STEM}-v1.json", dict(cue_sha256=cue_sha, model_sha256=MODEL_SHA, source_plan_sha256=sha(plan),
         source_archive_sha256=sha(waves), matcher_sha256=sha(matcher.ROOT / "portable/c161_cue_suffix_v1.h"), records=rows))


if __name__ == "__main__":
    main()
