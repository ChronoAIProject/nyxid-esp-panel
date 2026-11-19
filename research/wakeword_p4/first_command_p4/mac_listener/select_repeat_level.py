"""Choose the strongest equal-RMS early repeat that preserves both dry v5 readings."""
from pathlib import Path
import re

import numpy as np
import soundfile as sf

import check_repeat_p4 as replay

ch = replay.ch
STEM = replay.repeat.source.STEM + "-equal-rms"
PLAN = ch.RESULTS / (STEM + "-fit.json")
WAV = ch.RESULTS / "hye-nyx-v5-early-repeat-equal-rms.wav"
GAINS = (1.0, 2**-.5, .5, 2**-1.5, .25, .125)


def letters(text):
    return re.sub("[^A-Z]", "", text.upper())


def main():
    frozen = replay.repeat.source
    original = replay.grid.fit.original()[1]
    extra = replay.repeat.tracks()["repeat_trained"].astype(float) - original
    norm = np.linalg.norm(original.astype(float))
    waves = {"v5": original}
    meta = {}
    for i, gain in enumerate(GAINS):
        raw = original + gain * extra
        scale = norm / np.linalg.norm(raw)
        out = np.rint(raw * scale)
        assert np.max(abs(out)) <= 32767
        out = out.astype("<i2")
        delta_db = 20 * np.log10(np.linalg.norm(out.astype(float)) / norm)
        assert abs(delta_db) < .0001
        key = str(i)
        waves[key] = out
        meta[key] = dict(early_amplitude=gain, global_scale=scale, rms_delta_db=delta_db,
                         peak=int(np.abs(out.astype(int)).max()), pcm_sha256=ch.digest(out))
    reader, asr = ch.load_reader()
    readings = replay.reading(waves, reader)
    target = {k:letters(v) for k,v in readings["v5"].items()}
    assert all(v == "HEYNIX" for v in target.values())
    selected = None
    for key, r in meta.items():
        r["readings"] = readings[key]
        r["matches_v5"] = all(letters(v) == target[k] for k,v in readings[key].items())
        if selected is None and r["matches_v5"]:
            selected = key
    identity = {}
    if selected is not None:
        assert not WAV.exists()
        sf.write(WAV, waves[selected], 16000, subtype="PCM_16")
        identity = dict(file=WAV.name, sha256=ch.sha(WAV), samples=len(original))
    ch.save(PLAN, dict(records=meta, selected=selected, audio=identity, baseline_readings=readings["v5"], asr=asr,
                       sources={p.name:ch.sha(p) for p in (frozen.PLAN, frozen.FILES["trained"],
                           replay.grid.fit.fixed.CUE, replay.grid.fit.fixed.SOURCE_INFO, Path(__file__))},
                       policy="Scale only the extra early trained markers, then match total PCM RMS to original v5. "
                       "Pick the strongest tested repeat whose dry small.en and wav2vec2 greedy transcript letters "
                       "both equal the original. Selection uses only the dry tracks and no room scores."))


if __name__ == "__main__":
    main()
