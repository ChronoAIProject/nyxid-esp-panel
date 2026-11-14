"""Describe already-saved native training-witness probabilities; no scoring or inference."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import struct

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    lock_path = a.packet_root / "receipts/4090-fp32-internal-fullstream-execution-v1.json"
    lock = json.loads(lock_path.read_text())
    native_path = a.native_root / "report.json"
    native = json.loads(native_path.read_text())
    assert native["all_passed"] and len(native["cases"]) == 540
    rows, groups = [], defaultdict(list)
    for case in native["cases"]:
        model, index, steps = case["model"], case["index"], case["steps_per_repeat"]
        request = lock["requests"][index]
        original = (a.native_root / (model + f"-{index:03}") / "outputs.bin").read_bytes()
        assert sha(original) == case["outputs_sha256"] and len(original) == steps * 8
        assert original[:steps * 4] == original[steps * 4:]
        values = [v[0] for v in struct.iter_unpack("<f", original[:steps * 4])]
        with np.load(a.packet_root / "arrays" / (model + f"-{index:03}.npz"), allow_pickle=False) as z:
            clock = z["sample_end"].tolist()
        assert len(clock) == len(values)
        lead = [v for c, v in zip(clock, values, strict=True) if c < request["lead_samples"]]
        source = [v for c, v in zip(clock, values, strict=True) if c >= request["lead_samples"]]
        assert lead and source
        row = dict(model=model, request_index=index, request_order=request["order"],
                   source_target=request["source"]["target"], engine=request["source"]["engine"],
                   complete_stream_peak=max(values), pre_source_peak=max(lead), after_source_peak=max(source))
        rows.append(row)
        groups[model, row["source_target"], row["engine"]].append(row)
    summaries = []
    for (model, target, engine), values in sorted(groups.items()):
        summaries.append(dict(model=model, source_target=target, engine=engine, clips=len(values),
                              after_source_peak_min=min(v["after_source_peak"] for v in values),
                              after_source_peak_median=statistics.median(v["after_source_peak"] for v in values),
                              after_source_peak_max=max(v["after_source_peak"] for v in values),
                              pre_source_peak_max=max(v["pre_source_peak"] for v in values)))
    report = dict(schema="mac-fp32-original-training-witness-descriptive-v1", summaries=summaries, cases=rows,
                  original_native_result_sha256=sha(native_path.read_bytes()), original_gpu_lock_sha256=sha(lock_path.read_bytes()),
                  source_sha256=sha(Path(__file__).read_bytes()), original_shared_requests=135,
                  model_invocations=0, audio_decodes=0, threshold_selection=False,
                  limits="Descriptive peaks of already-consumed training witnesses, using original saved probability bytes/clocks and source-onset metadata. No event policy, recognition/FAR/latency scoring, operating-point selection, independent generalization/acceptance, training or new inference. High training-witness response does not establish held-out recognition quality.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
