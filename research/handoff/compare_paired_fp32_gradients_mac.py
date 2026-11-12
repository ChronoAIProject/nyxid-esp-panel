"""Compare received GPU CPU/CUDA first-batch arrays to independent Mac replay.

No model rerun, coefficient change, tolerance fit or original-failure revision.
Use the original numerical comparison's fixed atol=rtol=1e-5; save every result
including mismatches rather than treating cross-runtime similarity as universal.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("received", "mac-replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output required"
    root = args.received
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for row in manifest["files"]:
        data = (root / row["name"]).read_bytes()
        assert len(data) == row["size_bytes"] and sha(data) == row["sha256"]
    comparison = json.loads((root / "comparison.json").read_bytes())
    assert comparison["atol"] == comparison["rtol"] == 1e-5
    mac_bytes = (args.mac_replay / "report.json").read_bytes()
    mac = json.loads(mac_bytes)
    assert mac["original_probe_sha256"] == "0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f"
    assert sha((args.mac_replay / "mac-gradients.npz").read_bytes()) == mac["gradients_sha256"]
    with np.load(args.mac_replay / "mac-gradients.npz", allow_pickle=False) as data:
        mac_arrays = {k: data[k][0] for k in ("supervised", "consistency")}
    mac_arrays["metrics"] = np.array([mac["traces"][0][k]
                                      for k in ("main_bce", "pair_bce", "consistency")])
    values, reports = {"mac_independent_derivatives": mac_arrays}, {}
    for device in ("cpu", "cuda"):
        report = json.loads((root / device / "report.json").read_bytes())
        assert report["determinism_enabled"] and not report["tf32_enabled"]
        assert report["exact_repeat"] and report["no_weight_updates"]
        assert report["snapshot_sha256"] == mac["derived_h5_sha256"]
        assert report["batches_sha256"] == "7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb"
        assert sha((root / device / "arrays.npz").read_bytes()) == report["arrays_sha256"]
        reports[device] = report
        with np.load(root / device / "arrays.npz", allow_pickle=False) as data:
            assert set(data.files) == {"metrics", "supervised", "consistency"}
            values[device] = {k: data[k].astype(np.float64) for k in data.files}
    results = {}
    for left, right in (("cpu", "cuda"), ("mac_independent_derivatives", "cpu"),
                        ("mac_independent_derivatives", "cuda")):
        pairs = results[left + "_vs_" + right] = {}
        for key in ("metrics", "supervised", "consistency"):
            actual, expected = [values[name][key].astype(np.float64) for name in (left, right)]
            assert actual.shape == expected.shape and all(np.isfinite(v).all() for v in (actual, expected))
            delta = np.abs(actual - expected)
            bound = 1e-5 + 1e-5 * np.abs(expected)
            pairs[key] = dict(elements=actual.size, mismatched_elements=int(np.count_nonzero(delta > bound)),
                              passed=bool(np.all(delta <= bound)), maximum_absolute_error=float(delta.max()),
                              maximum_fraction_of_fixed_bound=float((delta / bound).max()),
                              relative_l2_error=float(np.linalg.norm(delta) / np.linalg.norm(expected)))
    result = dict(scope=__doc__, atol=1e-5, rtol=1e-5,
                  received_manifest_sha256=sha(manifest_bytes), mac_report_sha256=sha(mac_bytes),
                  same_certified_numerical_initialization_and_actual_first_batch=True,
                  comparisons=results, all_passed=all(v["passed"] for r in results.values() for v in r.values()),
                  script_sha256=sha(Path(__file__).read_bytes()),
                  limits="First saved batch only, Mac uses NumPy output derivatives plus CPU encoder "
                         "autodiff while GPU reports use TF full autodiff. No all50 new-policy comparison, "
                         "universal export equivalence, original-probe/v1-smoke passage, coefficient change, "
                         "training improvement, Hi ESP or P4 acceptance claim.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
