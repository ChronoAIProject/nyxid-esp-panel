"""Check primary-path retention on the three existing isolated synthetic speech cohorts."""
import argparse
from pathlib import Path

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from collect_multispeaker_formal_v1 import predict
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, frontend, read, save, sha
from replay_c161_history_gain_cohorts_v1 import REFERENCE
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-signed-confuser"


def main(parent=False):
    fit = read(RESULTS / f"{STEM}-training-v1.json")
    model = RESULTS / fit["reports"][0]["path"]
    assert sha(model) == fit["reports"][0]["sha256"]
    name = "signed"
    report_stem = STEM
    if parent:
        model = RESULTS / "4090-c161-paired-event-readout-speech-v1.tflite"
        assert sha(model) == fit["parent_sha256"]
        name = "parent"
        report_stem = "4090-c161-confuser-parent"
    for cohort in ("primed", "jenny", "melo"):
        spec = read(REFERENCE)["cohorts"][cohort]
        report = Path(spec["path"])
        assert sha(report) == spec["sha256"]
        rows, arrays = [], {}
        for ref in read(report)["records"]:
            if ref["phase"] != "synthetic":
                continue
            s = ref["source"]
            values, clock = frontend(load_wav(s))
            trace = Path(ref["trace"]["path"])
            assert sha(trace) == ref["trace"]["sha256"]
            with np.load(trace) as original:
                np.testing.assert_array_equal(clock, original["relative_block_return"])
            _, codes, meta = predict(values, str(model), REF_KERNEL)
            assert meta["quantized_input_sha256"] == ref["timing"]["quantized_input_sha256"]["c161"]
            key = str(ref["index"])
            arrays[key] = codes
            rows.append(dict(index=ref["index"], voice=s.get("voice") or s["voice_id"], text=s["text"],
                             label=s.get("target_id") or s["phrase_id"], carrier=s["carrier"] or "bare",
                             variant=s["variant"], pcm_sha256=s["pcm_sha256"], trace_sha256=sha(trace),
                             original=times(ref["events"]["c161"]), **{name: times(decode(codes, clock))}))
        dest = RESULTS / f"{report_stem}-isolated-{cohort}-scores-v1.npz"
        with dest.open("xb") as f:
            np.savez_compressed(f, **arrays)
        save(RESULTS / f"{report_stem}-isolated-{cohort}-v1.json", dict(model_sha256=sha(model),
             source_report_sha256=sha(report), scores_sha256=sha(dest), records=rows))
        print(cohort, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", action="store_true")
    main(parser.parse_args().parent)
