"""Authenticate and independently reduce the original rejected 540 saved cases.

No model, audio, frontend, inference, event detector or training code is loaded.
The two GPU reductions are checked against the authenticated saved arrays.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

import numpy as np

PACKET_REF = "57f7adf2529733f866f0e75b5e4690de83aee8a3"
MATCHED_REF = "0acd924da50964c3d710e2edb74c26075e757754"
ARCHIVE_SHA = "99228ac3c5d36a81a2ac231df18c3bf14c3ffdb4c6f144b3de78f0f5a87e8a20"
R = "research/wakeword_p4/results/"
B = R + "4090-hey-broad-oneway-consistency-"
REQUEST_PLAN = R + "4090-hey-broad-trained-quantization-plan-v1.json"
TARGETS = ("hey_willow", "hey_nyx")
CONDITIONS = ("twoway_consistency", "oneway_consistency")
KEYS = {"raw", "quantized_input", "float", "uint8", "float_repeat",
        "uint8_repeat", "offline_phase", "offline_nominal",
        "support_unclamped", "sample_end"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(repo, name, ref=PACKET_REF):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{name}"])


def stats(array):
    return dict(mean=float(np.mean(array)), p99=float(np.quantile(array, .99)),
                maximum=float(np.max(array)))


def array_identity(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=sha(array.tobytes()))


def audit(repo, archive):
    check(np.__version__ == "2.4.6", "Use original NumPy arithmetic version")
    data = archive.read_bytes()
    check(len(data) == 8546115 and sha(data) == ARCHIVE_SHA, "Original archive identity")
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        check(len(members) == 554, "Exact original archive member count")
        for member in members:
            path = PurePosixPath(member.name)
            check(member.isfile() and not path.is_absolute() and ".." not in path.parts,
                  "Regular relative payload only")
            check(member.name not in payloads and 0 < member.size < 2_000_000,
                  "Unique bounded payload")
            payloads[member.name] = tar.extractfile(member).read()
    manifest_bytes = blob(repo, R + "gpu-oneway-full540-saved-arrays-mac-v1-manifest.json")
    check(payloads.pop("manifest.json") == manifest_bytes, "Published manifest byte identity")
    manifest = json.loads(manifest_bytes)
    check(manifest["arrays"] == 540 and manifest["receipts"] == 13, "Original payload scope")
    inventory = {item["path"]: item for item in manifest["files"]}
    check(len(inventory) == 553 and set(inventory) == set(payloads), "Exact member names")
    for name, original in inventory.items():
        check(len(payloads[name]) == original["size"] and
              sha(payloads[name]) == original["sha256"], f"Payload identity: {name}")
        if name.startswith(R):
            check(payloads[name] == blob(repo, name), f"Original committed receipt: {name}")

    result = json.loads(payloads[B + "fixed-grid-screen-result-v1.json"])
    execution_bytes = payloads[B + "fixed-grid-screen-execution-v1.json"]
    execution = json.loads(execution_bytes)
    check(result["execution_sha256"] == sha(execution_bytes), "Screen execution binding")
    public_locks = {}
    for name, expected in execution["files"].items():
        if name.startswith("research/"):
            value = payloads[name] if name in payloads else blob(repo, name)
            check(sha(value) == expected, f"Original public source/receipt lock: {name}")
            public_locks[name] = expected
    check(REQUEST_PLAN in public_locks, "Original request plan is execution-locked")
    original_plan = json.loads(blob(repo, REQUEST_PLAN))
    rows = {row["order"]: row for row in original_plan["requests"]}
    check(len(rows) == len(original_plan["requests"]) == 135, "Original complete request axis")
    gates = original_plan["numerical_gates"]
    check(gates == dict(float_offline_stream_max_abs=.0001,
                       per_input_five_step_mean_abs_error_max=.02,
                       per_input_five_step_p99_abs_error_max=.1,
                       per_input_five_step_max_abs_error_max=.25), "Unchanged numerical gates")
    compiled = json.loads(payloads[B + "fixed-grid-compile-result-v1.json"])
    check(compiled["plan_sha256"] == sha(payloads[B + "fixed-grid-compile-plan-v1.json"]),
          "Compile plan binding")
    check(execution["exports"] == compiled["arms"], "Exact compiled exports in screen lock")
    formal = json.loads(payloads[B + "formal-result-v1.json"])
    check(formal["execution_sha256"] == sha(payloads[B + "formal-execution-v1.json"]),
          "Formal execution binding")
    weights = {(w["target"], w["condition"]): w for w in execution["weights"]}
    arms = {(a["target"], a["condition"]): a for a in compiled["arms"]}
    trained = {(c["target"], c["condition"]): c for c in formal["cases"]}
    axes = {(t, c) for t in TARGETS for c in CONDITIONS}
    check(set(weights) == set(arms) == set(trained) == axes, "Exact four model identities")
    model_bindings = []
    for key in sorted(axes):
        arm, weight, case = arms[key], weights[key], trained[key]
        final = case["final_repeat"]
        check(case["steps"] == 10000 and case["formal_repeats"] == 1,
              "Original final10000, one formal training repeat")
        check(final["weights_path"] == weight["path"] and
              final["weights_sha256"] == execution["files"][weight["path"]] == arm["final_h5_sha256"],
              "Formal weights -> compile -> screen metadata chain")
        check(arm["sha256"] == execution["files"][arm["path"]], "Screen export hash binding")
        model_bindings.append(dict(target=key[0], condition=key[1],
                                   h5_sha256=final["weights_sha256"], tflite_sha256=arm["sha256"]))

    expected_keys = {(t, c, order) for t, c in axes for order in rows}
    cases = {(c["target"], c["condition"], c["request_order"]): c for c in result["cases"]}
    check(len(result["cases"]) == len(cases) == 540 and set(cases) == expected_keys,
          "Complete original Cartesian product, no duplicates or filtering")
    identities, recomputed, tails = {}, {}, {}
    schema, total_steps, alternative_max = None, 0, 0.0
    summary = {}
    for key, case in cases.items():
        target, condition, order = key
        row = rows[order]
        index = list(rows).index(order)
        name = f".local/wakeword/hey-broad-oneway-consistency-fixed-grid-screen-v1/{target}-{condition}-{index:03}.npz"
        check(case["arrays_path"] == name and sha(payloads[name]) == case["arrays_sha256"],
              "Case identity -> original array bytes")
        with np.load(io.BytesIO(payloads[name]), allow_pickle=False) as npz:
            check(set(npz.files) == KEYS and len(npz.files) == len(KEYS), "Exact NPZ array schema")
            a = {name: npz[name] for name in KEYS}
        raw, floating, integer = a["raw"], a["float"], a["uint8"]
        check(raw.dtype == np.uint16 and raw.ndim == 2 and raw.shape[1] == 40,
              "Original uint16 frontend shape")
        check(len(raw) == (row["samples"] + 159) // 160 - 2, "Full original frontend frame count")
        n = (len(raw) + 2) // 3
        expected_dtypes = {"raw": "uint16", "quantized_input": "int8", "float": "float64",
                           "uint8": "uint8", "float_repeat": "float32", "uint8_repeat": "uint8",
                           "offline_phase": "float64", "offline_nominal": "float64",
                           "support_unclamped": "int64", "sample_end": "int64"}
        for field, value in a.items():
            check(value.dtype == np.dtype(expected_dtypes[field]) and np.isfinite(value).all(),
                  f"Original finite typed array: {field}")
            expected_shape = (len(raw), 40) if field == "raw" else (
                (n * 3, 40) if field == "quantized_input" else (
                    (n - 74,) if field.startswith("offline_") else (n,)))
            check(value.shape == expected_shape, f"Complete original array length: {field}")
        current_schema = {name: value.dtype.str for name, value in a.items()}
        if schema is None:
            schema = current_schema
        check(current_schema == schema, "Uniform dtype schema")
        check(np.all((floating >= 0) & (floating <= 1)) and
              np.array_equal(floating, floating.astype(np.float32).astype(np.float64)),
              "Original losslessly promoted float32 probabilities")
        repeats = (floating.tobytes() == a["float_repeat"].astype(np.float64).tobytes() and
                   integer.tobytes() == a["uint8_repeat"].tobytes())
        check(repeats == case["repeat_exact"], "Actual saved repeat equality")
        ident = array_identity(raw)
        if order in identities:
            check(identities[order] == ident, "Same complete raw feature bytes across all four models")
        identities[order] = ident
        values = np.minimum(np.pad(raw, ((0, n * 3 - len(raw)), (0, 0))).astype(np.float32) / 25.6, 26)
        scale, zero = case["input_quantization"]
        oscale, ozero = case["output_quantization"]
        tensors = {v["index"]: v for v in arms[(target, condition)]["tensors"]}
        output_index = arms[(target, condition)]["ops"][-1]["outputs"][0]
        check([tensors[0]["scale"], tensors[0]["zero"]] == [scale, zero] and
              [tensors[output_index]["scale"], tensors[output_index]["zero"]] == [oscale, ozero],
              "Quantization metadata from exact compiled-arm receipt")
        quantized = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
        check(quantized.tobytes() == a["quantized_input"].tobytes(), "All quantized input bytes")
        clocks = np.arange(1, n + 1, dtype=np.int64) * 480 + 320
        check(np.array_equal(clocks, a["support_unclamped"]) and
              np.array_equal(np.minimum(clocks, row["samples"]), a["sample_end"]),
              "Complete original unclamped and sample-end clocks")
        signed = (integer.astype(np.float64) - ozero) * oscale - floating
        smooth = np.abs(np.convolve(signed, np.full(5, .2), "valid"))
        # A second arithmetic form checks the convolution without fitting a tolerance.
        alternative = np.abs(np.lib.stride_tricks.sliding_window_view(signed, 5).mean(axis=1))
        delta = float(np.max(np.abs(alternative - smooth)))
        check(delta <= 1e-15, "Independent window arithmetic roundoff bound")
        alternative_max = max(alternative_max, delta)
        five, raw_stats = stats(smooth), stats(np.abs(signed))
        phase = float(np.max(np.abs(a["offline_phase"] - floating[74:])))
        check(five == case["smoothed_abs_error"] and raw_stats == case["raw_abs_error"] and
              phase == case["phase_max_error"], "All stored scalar statistics exactly reproduced")
        passed = bool(repeats and phase <= gates["float_offline_stream_max_abs"] and
                      five["mean"] <= gates["per_input_five_step_mean_abs_error_max"] and
                      five["p99"] <= gates["per_input_five_step_p99_abs_error_max"] and
                      five["maximum"] <= gates["per_input_five_step_max_abs_error_max"])
        check(passed == case["passed"], "Original gate decision")
        recomputed[key] = dict(passed=passed, stats=five)
        name = condition + "/" + target
        group = summary.setdefault(name, dict(cases=0, passed=0, failed_requests=[],
                                             worst_mean=0, worst_p99=0, worst_maximum=0))
        group["cases"] += 1
        group["passed"] += int(passed)
        if not passed:
            group["failed_requests"].append(order)
            indices = np.flatnonzero(smooth > .1)
            tails[key] = dict(condition=condition, request_order=order, phase_max=phase,
                              five_stats=five, windows_above_point1=int(len(indices)),
                              windows_above_point1_after_startup=int(np.sum(indices >= 74)),
                              largest_window_start=int(np.argmax(smooth)))
        for metric, value in five.items():
            group["worst_" + metric] = max(group["worst_" + metric], value)
        total_steps += n

    original_audit = json.loads(payloads[B + "fixed-grid-screen-independent-audit-v2.json"])
    check(summary == original_audit["summary"], "GPU audit v2: all 540 summaries exactly reproduced")
    check(not result["all_passed"] and original_audit["qualified_complete270_conditions"] == [],
          "Neither complete condition qualified")
    matched = json.loads(blob(repo, R + "4090-oneway-matched-failure-reduction-v1.json", MATCHED_REF))
    check(matched["source_sha256"] == sha(payloads[B + "fixed-grid-screen-result-v1.json"]),
          "Matched reduction uses exact original result")
    transition_counts = {}
    for original in matched["targets"]:
        target = original["target"]
        transitions, differences, failures = {}, {m: [] for m in ("mean", "p99", "maximum")}, []
        for order in rows:
            control = recomputed[(target, CONDITIONS[0], order)]
            treatment = recomputed[(target, CONDITIONS[1], order)]
            transition = ("pass" if control["passed"] else "fail") + "->" + ("pass" if treatment["passed"] else "fail")
            transitions.setdefault(transition, []).append(order)
            for metric in differences:
                differences[metric].append(treatment["stats"][metric] - control["stats"][metric])
            failures += [tails[(target, c, order)] for c in CONDITIONS if (target, c, order) in tails]
        signed_stats = {}
        for metric, values in differences.items():
            values = np.asarray(values)
            signed_stats[metric] = dict(improved=int(np.sum(values < 0)), equal=int(np.sum(values == 0)),
                                        worsened=int(np.sum(values > 0)), minimum=float(values.min()),
                                        median=float(np.median(values)), maximum=float(values.max()))
        check(transitions == original["all135_transitions"] and
              signed_stats == original["all135_signed_treatment_minus_control"] and
              failures == original["all_failed_cases"], "Complete matched reduction including every failure")
        transition_counts[target] = {name: len(orders) for name, orders in transitions.items()}
    check(b"KeyError" in payloads[B + "screen-original-audit-failure-v1.log"],
          "Original failed audit retained")
    return dict(schema="mac-oneway-saved540-cross-audit-v1", archive_sha256=ARCHIVE_SHA,
                archive_bytes=len(data), exact_archive_members=554, authenticated_payloads=553,
                source_ref=PACKET_REF, matched_ref=MATCHED_REF,
                reader_sha256=sha(Path(__file__).read_bytes()), python=sys.version.split()[0],
                numpy=np.__version__, original_public_locks=public_locks,
                model_metadata_bindings=model_bindings, full_cases=540,
                unique_raw_inputs=len(identities), full_steps_across_540=total_steps,
                array_dtype_schema=schema, numerical_gates=gates,
                max_secondary_window_arithmetic_delta=alternative_max,
                gpu_v2_summary_exact=True, gpu_matched_reduction_exact=True,
                summary=summary, matched_transitions=transition_counts,
                failed_cases=len(tails), failed_cases_with_errors_after_startup=sum(
                    row["windows_above_point1_after_startup"] > 0 for row in tails.values()),
                qualified_complete270_conditions=[], original_failure_retained=True,
                model_invocations=0, media_reads=0,
                limits="Saved numerical arrays only. Model/audio/private training assets are not in this packet; their bytes or origin inference are not independently reproduced. Metadata hash-chain checked. No new recognition, FAR, model selection, fresh development, acceptance, P4 or deployment claim. Original rejection unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    check(not args.report.exists(), "Refuse overwriting an original report")
    result = audit(args.repository_root.resolve(), args.archive)
    with args.report.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "full_cases", "full_steps_across_540", "summary", "matched_transitions",
        "gpu_v2_summary_exact", "gpu_matched_reduction_exact", "qualified_complete270_conditions")}, indent=2))


if __name__ == "__main__":
    main()
