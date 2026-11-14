"""Reduce the trained fullstream UART protocol; does not establish capture origin."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


BASE = Path("research/wakeword_p4")
TOLERANCE = 1e-4
DEADLINE_US = 30000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def strict_error(actual, expected):
    # Promote each original float32 value before subtraction. Never gate on
    # the device's float32-rounded subtraction or a printed summary alone.
    assert math.isfinite(actual) and math.isfinite(expected)
    assert 0 <= actual <= 1 and 0 <= expected <= 1
    error = abs(actual - expected)
    assert error <= TOLERANCE
    return error


def exact_keys(obj, names):
    assert set(obj) == set(names)


def integer(obj, key, minimum=0, maximum=None):
    value = obj[key]
    assert type(value) is int and value >= minimum
    if maximum is not None:
        assert value <= maximum
    return value


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        assert key not in out, key
        out[key] = value
    return out


def reject_constant(value):
    raise AssertionError("Non-finite JSON number: " + value)


def load_fixtures(repo, run):
    plan_bytes = (run / "fixture-plan.json").read_bytes()
    assert plan_bytes == (repo / BASE / "results/mac-fp32-trained-fullstream-p4-preparation-v1.json").read_bytes()
    plan = json.loads(plan_bytes)
    assert plan["schema"] == "mac-fp32-trained-fullstream-p4-preparation-v1"
    assert plan["repeats"] == 2 and plan["expected_invocations"] == 8336
    assert plan["max_abs"] == TOLERANCE and plan["every_invoke_deadline_us"] == DEADLINE_US
    assert sha((repo / BASE / "experiments/fp32_trained_fullstream_p4_v1.cpp").read_bytes()) == plan["probe_sha256"]
    for path, wanted in plan["generated_files"].items():
        data = (run / path).read_bytes()
        assert len(data) == wanted["bytes"] and sha(data) == wanted["sha256"]
    link_bytes = (repo / BASE / "results/mac-fp32-trained-fullstream-p4-link-v1.json").read_bytes()
    link = json.loads(link_bytes)
    assert link["preparation_sha256"] == sha(plan_bytes)
    assert link["all_embedded_assets_byte_exact"] and link["runtime_components_exact"]
    image = (run / "build/nyx_trained_fp32_probe.bin").read_bytes()
    assert len(image) == link["image_bytes"] and sha(image) == link["image_sha256"]
    references = []
    for number, case in enumerate(plan["cases"]):
        assert case["case"] == number
        tf_data = (run / case["tf_file"]).read_bytes()
        lite_data = (run / case["litert_file"]).read_bytes()
        assert sha(tf_data) == case["tf_sha256"] and sha(lite_data) == case["litert_sha256"]
        assert len(tf_data) == len(lite_data) == 4 * case["steps"]
        refs = [struct.unpack("<" + "f" * case["steps"], data) for data in (tf_data, lite_data)]
        assert all(math.isfinite(x) and 0 <= x <= 1 for values in refs for x in values)
        references.append(dict(case=number, steps=case["steps"], tf=refs[0], litert=refs[1]))
    assert len(references) == 24 and sum(r["steps"] for r in references) * 2 == 8336
    binding = dict(preparation_sha256=sha(plan_bytes), build_report_sha256=sha(link_bytes),
                   image_sha256=sha(image), original_execution_sha256=plan["original_execution_sha256"],
                   original_packet_manifest_sha256=plan["original_packet_manifest_sha256"])
    return references, binding


def protocol_records(raw):
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("P4_TRAINED_"):
            assert "P4_TRAINED_" not in line, "Malformed protocol prefix"
            continue
        kind, payload = line.split(" ", 1)
        assert kind in ("P4_TRAINED_BEGIN", "P4_TRAINED_STEP", "P4_TRAINED_CASE", "P4_TRAINED_DONE")
        data = json.loads(payload, object_pairs_hook=unique_object, parse_constant=reject_constant)
        assert isinstance(data, dict)
        yield kind, data


def reduce_capture(raw, references):
    records = iter(protocol_records(raw))
    kind, begin = next(records)
    assert kind == "P4_TRAINED_BEGIN"
    exact_keys(begin, ("cases", "repeats", "configured_cpu_mhz", "arena_bytes", "variable_bytes"))
    assert integer(begin, "cases") == len(references) == 24
    assert integer(begin, "repeats") == 2
    assert integer(begin, "configured_cpu_mhz") == 360
    assert integer(begin, "arena_bytes") == 256 * 1024
    assert integer(begin, "variable_bytes") == 64 * 1024
    results, invokes, worst_tf, worst_lite, worst_us = [], 0, 0., 0., 0
    for case in references:
        original_bits = None
        original_arenas = None
        for repeat in range(2):
            bits, durations, errors_tf, errors_lite = [], [], [], []
            for step in range(case["steps"]):
                kind, rec = next(records)
                assert kind == "P4_TRAINED_STEP"
                exact_keys(rec, ("case", "repeat", "step", "bits", "us"))
                assert integer(rec, "case") == case["case"]
                assert integer(rec, "repeat") == repeat and integer(rec, "step") == step
                assert isinstance(rec["bits"], str) and re.fullmatch(r"[0-9a-f]{8}", rec["bits"])
                value = struct.unpack("<f", int(rec["bits"], 16).to_bytes(4, "little"))[0]
                errors_tf.append(strict_error(value, case["tf"][step]))
                errors_lite.append(strict_error(value, case["litert"][step]))
                durations.append(integer(rec, "us", 1, DEADLINE_US))
                bits.append(rec["bits"])
            kind, summary = next(records)
            assert kind == "P4_TRAINED_CASE"
            exact_keys(summary, ("case", "repeat", "steps", "passed", "max_abs_tf", "max_us", "setup_us", "arena_used", "variable_used"))
            assert integer(summary, "case") == case["case"] and integer(summary, "repeat") == repeat
            assert integer(summary, "steps") == case["steps"] and summary["passed"] is True
            max_tf, max_lite, max_us = max(errors_tf), max(errors_lite), max(durations)
            # Bind the printed float32-rounded %.9g summary to the raw outputs;
            # the stricter float64 gates have already checked every output.
            assert type(summary["max_abs_tf"]) in (int, float) and math.isfinite(summary["max_abs_tf"])
            assert summary["max_abs_tf"] == float(format(max(f32(x) for x in errors_tf), ".9g"))
            assert integer(summary, "max_us") == max_us
            setup_us = integer(summary, "setup_us", 1)
            arenas = (integer(summary, "arena_used", 1, begin["arena_bytes"]),
                      integer(summary, "variable_used", 1, begin["variable_bytes"]))
            if repeat == 0:
                original_bits, original_arenas = bits, arenas
            else:
                assert bits == original_bits and arenas == original_arenas
            invokes += case["steps"]
            worst_tf, worst_lite, worst_us = max(worst_tf, max_tf), max(worst_lite, max_lite), max(worst_us, max_us)
            output_bytes = b"".join(int(b, 16).to_bytes(4, "little") for b in bits)
            results.append(dict(case=case["case"], repeat=repeat, steps=case["steps"],
                                output_sha256=sha(output_bytes), max_abs_tf_float64=max_tf,
                                max_abs_litert_float64=max_lite, every_invoke_deadline_pass=True,
                                max_invoke_us=max_us, first_invoke_us=durations[0], setup_us=setup_us,
                                arena_used=arenas[0], variable_used=arenas[1]))
    kind, done = next(records)
    assert kind == "P4_TRAINED_DONE"
    exact_keys(done, ("passed", "cases", "repeats", "stack_free"))
    assert done["passed"] is True and integer(done, "cases") == 24 and integer(done, "repeats") == 2
    integer(done, "stack_free", 1)
    assert next(records, None) is None and invokes == 8336
    return dict(executions=len(results), invocations=invokes,
                max_abs_tf_float64=worst_tf, max_abs_litert_float64=worst_lite,
                max_invoke_us=worst_us, every_invoke_deadline_pass=True,
                fresh_repeat_output_bytes_exact=True, case_reports=results,
                reported_stack_free=done["stack_free"], protocol_complete=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "run-root", "capture", "report"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.report.exists()
    references, binding = load_fixtures(a.repository_root, a.run_root)
    raw = a.capture.read_bytes()
    reduction = reduce_capture(raw, references)
    result = dict(schema="mac-trained-fp32-p4-saved-protocol-reduction-v1",
                  source_sha256=sha(Path(__file__).read_bytes()), capture_sha256=sha(raw),
                  binding=binding, reduction=reduction,
                  hardware_origin_independently_established=False, candidate_qualified=False,
                  limits="Saved protocol numeric/byte/timing validation only. Capture origin must separately be bound to the actual board/slot/flashed image by the app-only execution and restore receipts. Not all540, full-panel headroom, recognition/FAR, audio or long-call acceptance.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in reduction.items() if k != "case_reports"}))


if __name__ == "__main__":
    main()
