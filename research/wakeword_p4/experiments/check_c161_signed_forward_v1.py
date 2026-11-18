"""Compare the frozen signed readout, C161 and official Willow on new matched requests."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

import c161_four_voice_forward_v1 as forward
from audit_c161_history_paths_v1 import arbitrate, decode, times
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-signed-forward"
OUT = Path(".local/wakeword/c161-signed-forward-v1")
INPUTS = RESULTS / f"{STEM}-plan-v1.json"
PLAN = RESULTS / f"{STEM}-execution-v1.json"
VOICES = ("af_aoede", "am_eric", "bf_lily", "am_fenrir")
SCENES = (
    ("Dinner should be ready in ten minutes.", "Nick, the blue folder is yours.", "lower the fan speed.", 4000),
    ("I left my coat beside the stairs.", "Nicole, please bring six spoons.", "start the kitchen timer.", 8000),
    ("The last bus leaves before midnight.", "Kevin asked for six spare keys.", "read my shopping list.", 12000),
)


def bind():
    data = read(INPUTS)
    model = RESULTS / "4090-c161-signed-confuser-single-v1.tflite"
    implementation = (Path(__file__).resolve(), Path(forward.reference.__file__).resolve(), model)
    for path in implementation:
        data["files"][str(path)] = sha(path)
    data["inputs_sha256"] = sha(INPUTS)
    save(PLAN, data)


def score():
    fit = read(RESULTS / "4090-c161-signed-confuser-training-v1.json")
    candidate = RESULTS / fit["reports"][0]["path"]
    assert sha(candidate) == fit["reports"][0]["sha256"]
    reference = RESULTS / f"{STEM}-v1.json"
    report = read(reference)
    baseline = RESULTS / f"{STEM}-scores-v1.npz"
    assert sha(baseline) == report["scores_sha256"] and sha(PLAN) == report["plan_sha256"]
    native = binding()
    rows, arrays = [], {}
    with np.load(baseline) as prior, patch.object(live, "MODEL", candidate):
        for i, r in enumerate(report["records"]):
            pcm = load_wav(r)
            codes, clock, emitted, suppressed = live.run_stream(native, pcm)
            clock = np.asarray(clock)
            np.testing.assert_array_equal(clock, prior[f"{i}_clock"])
            arrays[str(i)] = np.asarray(codes, np.uint8)
            std, aux = [decode(v, clock) for v in arrays[str(i)]]
            assert arbitrate(std, aux) == (emitted, suppressed)
            rows.append(dict(key=i, voice=r["voice"], scene=r["scene"], role=r["target_id"],
                             pcm_sha256=r["pcm_sha256"], prefix_sha256=r["prefix_sha256"],
                             start=r["start"], end=r["end"], first_phone=r["first_phone"],
                             original=r["original_events"], original_pair=r["pair_events"],
                             willow=r["official_events"], signed=times(std), signed_pair=times(emitted)))
    output = RESULTS / f"{STEM}-candidate-scores-v1.npz"
    with output.open("xb") as f:
        np.savez_compressed(f, **arrays)
    save(RESULTS / f"{STEM}-candidate-v1.json", dict(reference_sha256=sha(reference), model_sha256=sha(candidate),
         scores_sha256=sha(output), records=rows))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "bind":
        bind()
        sys.exit(0)
    reference = forward.reference
    with patch.multiple(forward, PLAN=INPUTS, OUT=OUT, STEM=STEM, VOICES=VOICES, SCENES=SCENES), \
         patch.multiple(reference, PLAN=PLAN, CASES=OUT / "official", STEM=STEM, __file__=__file__):
        {"generate": forward.generate, "run": reference.execute, "case": reference.case,
         "summarize": reference.summarize, "score": score}[action]()
