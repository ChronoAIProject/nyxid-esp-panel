"""Place the fitted marker stems before the original fixed cue in the same 1.45 s WAV."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import fit_bella_multirate_stems as fitted

P, playback, native = fitted.P, fitted.playback, fitted.native
ROOT = fitted.gain_grid.ROOT
BASE = ROOT / "results/4090-bella-early-marker"
FIT = BASE.with_suffix(".json")
WAV = BASE.with_name("hye-nyx-af-bella-early-marker-experiment.wav")
PLAN = BASE.with_name(BASE.name + "-plan.json")
REPORT = BASE.with_name(BASE.name + "-wake.json")
WORK = Path(".local/wakeword/bella-early-marker")
PROTOCOL = ROOT / "experiments/bella_early_marker_protocol.md"


def build():
    original, rate = sf.read(playback.OUT.with_name(P.read(playback.OUT)["cue"]), dtype="int16")
    assert rate == 16000 and P.digest(original) == P.read(playback.OUT)["pcm_sha256"]
    fitted_wav, rate = sf.read(fitted.WAV, dtype="int16")
    spec = P.read(fitted.FIT)
    assert rate == 16000 and P.digest(fitted_wav) == spec["pcm_sha256"]
    stems = fitted.stems
    _, _, _, codes = stems.gains.fit.grid.fit.original()
    with ZipFile(stems.gains.PACK) as pack:
        source = json.loads(pack.read("manifest.json"))
        entry, = [r for r in source["records"] if r.get("wake_only", {}).get("sha256") == P.sha(stems.overlap.board.bella.CUE)]
    extra = np.zeros(len(original))
    for r, code, level in zip(spec["stems"], codes, source["gains"], strict=True):
        a, b = r["window"]
        part = fitted_wav[a:b].astype(float) - original[a:b]
        pos = r["start"] - a
        part[pos:pos+len(code)] += code * (level * entry["prefix_scale"] * 10**.25)
        extra[a-6400:b-6400] = part
    up, down = extra > 0, extra < 0
    base = original.astype(float)
    scale = min(1., float(np.min((32767-base[up])/extra[up])), float(np.min((-32768-base[down])/extra[down])))
    assert 0 < scale <= 1
    total = np.rint(base + scale*extra)
    assert total.min() >= -32768 and total.max() <= 32767
    cue = total.astype("<i2")
    assert np.array_equal(cue[spec["stems"][0]["window"][0]:], original[spec["stems"][0]["window"][0]:])
    blob = playback.playback.sources.wav(cue, "PCM_16")
    if WAV.exists():
        assert WAV.read_bytes() == blob
    else:
        WAV.write_bytes(blob)
    sources = (Path(__file__), PROTOCOL, fitted.WAV, fitted.FIT, playback.OUT, stems.gains.PACK)
    P.save(FIT, dict(cue=WAV.name, cue_sha256=P.sha(WAV), pcm_sha256=P.digest(cue), samples=len(cue), rate=16000,
                     added_stem_scale=scale, advanced_samples=6400, changed_samples=int(np.count_nonzero(original!=cue)),
                     peak=int(np.max(np.abs(total))), sources={str(p): P.sha(p) for p in sources}))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage == "dual":
        native.PLAN, native.WORK = PLAN, WORK
        native.dual()
    else:
        fitted.PLAN, fitted.WORK, fitted.WAV, fitted.FIT, fitted.REPORT = PLAN, WORK, WAV, FIT, REPORT
        {"prepare": fitted.prepare, "report": fitted.report}[stage]()
