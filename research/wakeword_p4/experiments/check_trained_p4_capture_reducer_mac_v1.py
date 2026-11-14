"""Constructed protocol corruption checks; never invoke a model or touch a board."""
import argparse
import copy
import json
from pathlib import Path
import struct

import reduce_fp32_trained_p4_capture_mac_v1 as reducer


def encode(records):
    return ("boot: constructed input, not hardware\n" + "".join(
        kind + " " + json.dumps(rec, separators=(",", ":")) + "\n" for kind, rec in records)).encode()


def construct(refs):
    records = [("P4_TRAINED_BEGIN", dict(cases=24, repeats=2, configured_cpu_mhz=360,
                                          arena_bytes=262144, variable_bytes=65536))]
    for ref in refs:
        for repeat in range(2):
            for step, value in enumerate(ref["tf"]):
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                records.append(("P4_TRAINED_STEP", dict(case=ref["case"], repeat=repeat, step=step,
                                                        bits=f"{bits:08x}", us=27000 if step == 0 else 1500 + step)))
            records.append(("P4_TRAINED_CASE", dict(case=ref["case"], repeat=repeat, steps=ref["steps"],
                                                    passed=True, max_abs_tf=0., max_us=27000, setup_us=1234,
                                                    arena_used=55120, variable_used=1040)))
    records.append(("P4_TRAINED_DONE", dict(passed=True, cases=24, repeats=2, stack_free=2048)))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "run-root", "output-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.output_dir.exists()
    a.output_dir.mkdir()
    refs, binding = reducer.load_fixtures(a.repository_root, a.run_root)
    records = construct(refs)
    raw = encode(records)
    (a.output_dir / "constructed-only-capture.log").write_bytes(raw)
    valid = reducer.reduce_capture(raw, refs)
    assert valid["executions"] == 48 and valid["invocations"] == 8336
    assert valid["max_abs_tf_float64"] == 0 and valid["max_invoke_us"] == 27000
    rejected = []

    def reject(name, data, references=refs):
        try:
            reducer.reduce_capture(data, references)
        except (AssertionError, StopIteration, ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError("Accepted invalid capture: " + name)

    def mutate(name, index, field, value):
        changed = copy.deepcopy(records)
        changed[index][1][field] = value
        reject(name, encode(changed))

    reject("missing_done", encode(records[:-1]))
    reject("missing_step", encode(records[:1] + records[2:]))
    reject("duplicate_step", encode(records[:2] + records[1:]))
    reject("duplicate_begin", encode(records[:1] + records))
    reject("duplicate_done", encode(records + records[-1:]))
    reject("reordered_steps", encode(records[:1] + [records[2], records[1]] + records[3:]))
    reject("reboot_midstream", encode(records[:20] + records[:1] + records[20:]))
    mutate("wrong_cpu", 0, "configured_cpu_mhz", 400)
    mutate("wrong_case_count", 0, "cases", 4)
    mutate("wrong_step", 1, "step", 1)
    mutate("boolean_timing", 1, "us", True)
    mutate("zero_timing", 1, "us", 0)
    mutate("slow_cold_invoke", 1, "us", 30001)
    mutate("nan_bits", 1, "bits", "7fc00000")
    mutate("negative_bits", 1, "bits", "bf800000")
    wrong_value = 0. if refs[0]["tf"][0] > .5 else 1.
    wrong_bits = struct.unpack("<I", struct.pack("<f", wrong_value))[0]
    mutate("numeric_failure_despite_passed_summaries", 1, "bits", f"{wrong_bits:08x}")
    first_summary = next(i for i, (k, _) in enumerate(records) if k == "P4_TRAINED_CASE")
    mutate("lying_max_timing", first_summary, "max_us", 26999)
    mutate("lying_float_summary", first_summary, "max_abs_tf", .00001)
    mutate("arena_over_capacity", first_summary, "arena_used", 262145)
    mutate("false_case_summary", first_summary, "passed", False)
    mutate("false_done", len(records) - 1, "passed", False)
    changed = copy.deepcopy(records)
    second_first = next(i for i, (k, r) in enumerate(changed) if k == "P4_TRAINED_STEP" and r["case"] == 0 and r["repeat"] == 1)
    second_summary = next(i for i, (k, r) in enumerate(changed) if k == "P4_TRAINED_CASE" and r["case"] == 0 and r["repeat"] == 1)
    old_bits = int(changed[second_first][1]["bits"], 16)
    new_bits = old_bits - 1 if refs[0]["tf"][0] >= 1. else old_bits + 1
    value = struct.unpack("<f", new_bits.to_bytes(4, "little"))[0]
    delta = reducer.strict_error(value, refs[0]["tf"][0])
    changed[second_first][1]["bits"] = f"{new_bits:08x}"
    changed[second_summary][1]["max_abs_tf"] = float(format(reducer.f32(delta), ".9g"))
    reject("one_ulp_fresh_repeat_difference_within_numeric_tolerance", encode(changed))
    changed_refs = copy.deepcopy(refs)
    changed_refs[0]["litert"] = (wrong_value,) + changed_refs[0]["litert"][1:]
    reject("litert_difference_even_when_tf_exact", raw, changed_refs)
    duplicate_json_key = raw.replace(b'"configured_cpu_mhz":360', b'"configured_cpu_mhz":400,"configured_cpu_mhz":360', 1)
    reject("duplicate_json_key", duplicate_json_key)
    reject("unknown_error_record", raw.replace(b"P4_TRAINED_STEP", b"P4_TRAINED_ERROR", 1))
    # Values straddle decimal1e-4 while a float32 subtraction rounds down to
    # the device tolerance. This must fail the independent promoted gate.
    small = reducer.f32(4e-12)
    large_bits = struct.unpack("<I", struct.pack("<f", 1e-4))[0] + 1
    large = struct.unpack("<f", large_bits.to_bytes(4, "little"))[0]
    assert large - small > 1e-4 and reducer.f32(large - small) <= reducer.f32(1e-4)
    try:
        reducer.strict_error(large, small)
    except AssertionError:
        rejected.append("float32_subtraction_hides_float64_tolerance_failure")
    else:
        raise AssertionError("Promoted tolerance gate accepted an over-bound error")
    report = dict(schema="mac-trained-p4-capture-reducer-constructed-checks-v1",
                  source_sha256=reducer.sha(Path(__file__).read_bytes()),
                  reducer_sha256=reducer.sha(Path(reducer.__file__).read_bytes()),
                  binding=binding, valid_constructed_executions=valid["executions"],
                  valid_constructed_invocations=valid["invocations"],
                  constructed_capture_sha256=reducer.sha(raw), negative_cases_rejected=rejected,
                  all_passed=True, actual_p4_invocations=0, hardware_accessed=False,
                  limits="Constructed protocol with original authenticated reference values and invented timings/resources. Validates reducer behavior only, not P4 numeric/timing/arena, new model inference, recognition or acceptance.")
    (a.output_dir / "constructed-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
