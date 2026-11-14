"""Check float event arithmetic using pinned original UINT8 policy and constructed inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types

import numpy as np

from float32_event_policy_mac_reference_v1 import event_steps, event_samples, EVENT_ARITHMETIC

PIN = "f7239057cabe25be1d7ee1269564e87d9baa6f60"
LEGACY_SHA = "6a35d8e591545881a42c5d3e4a8b0bac62a4940bbf042a1b01fa44049409653f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    old_bytes = subprocess.check_output(["git", "show", PIN + ":research/wakeword_p4/experiments/uint8_event_policy_v1.py"], cwd=a.repository_root)
    assert sha(old_bytes) == LEGACY_SHA
    legacy = types.ModuleType("pinned_original_uint8_policy")
    exec(compile(old_bytes, "pinned-original-uint8-policy", "exec"), legacy.__dict__)
    grid = sorted(set([.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994] + [.5 + i / 256 for i in range(128)]))
    assert len(grid) == 138
    rng = np.random.default_rng(20261114)
    sequences = [np.zeros(512, np.uint8), np.full(512, 255, np.uint8),
                 np.array(([255] * 5 + [0] * 67 + [255] * 5) * 5, np.uint8),
                 np.array([255] * 5 + [0] * 10 + [255] * 100, np.uint8)]
    sequences += [rng.integers(0, 256, size=512, dtype=np.uint8) for _ in range(6)]
    comparisons = 0
    for raw in sequences:
        expanded = raw.astype(np.float32) / 256.0
        before = expanded.tobytes()
        for threshold in grid:
            assert event_steps(expanded, threshold) == legacy.event_steps(raw, threshold)
            comparisons += 1
        assert expanded.tobytes() == before
    # Cover every attainable five-byte sum, independently of the event FSM.
    sums_checked = 0
    for total in range(1276):
        quotient, remainder = divmod(total, 255)
        window = [255] * quotient + ([remainder] if remainder else [])
        window += [0] * (5 - len(window))
        assert len(window) == 5 and sum(window) == total
        assert math.fsum(float(np.float32(x / 256.0)) for x in window) / 5.0 == total / 1280.0
        sums_checked += 1
    known = [
        ("full window required", [1] * 4, []),
        ("first complete window", [1] * 5, [5]),
        ("high plateau does not retrigger", [1] * 200, [5]),
        ("cooldown then low rearm", [1] * 5 + [0] * 67 + [1] * 5, [5, 75]),
        ("low only during cooldown cannot rearm", [1] * 5 + [0] * 10 + [1] * 100, [5]),
    ]
    for name, values, expected in known:
        assert event_steps(np.array(values, np.float32), .5) == expected, name
    probability = np.float32(.501)
    raw = np.full(5, probability, np.float32)
    threshold = float(probability)
    assert event_steps(raw, threshold) == [5]
    rounded = np.rint(raw.astype(np.float64) * 256).astype(np.uint8)
    assert legacy.event_steps(rounded, threshold) == []
    assert event_steps(np.ones(5, np.float32), 1.0) == [5]
    assert event_samples(np.ones(5, np.float32), .5, 2600) == [(5, 2600)]
    rejected = []
    invalid = [
        ("float64 vector", np.ones(5, np.float64), .5),
        ("UINT8 vector", np.ones(5, np.uint8), .5),
        ("implicit conversion list", [1.] * 5, .5),
        ("two-dimensional vector", np.ones((1, 5), np.float32), .5),
        ("NaN probability", np.array([np.nan], np.float32), .5),
        ("infinite probability", np.array([np.inf], np.float32), .5),
        ("negative probability", np.array([-.001], np.float32), .5),
        ("probability above one", np.array([np.nextafter(np.float32(1), np.float32(2))], np.float32), .5),
        ("zero threshold", np.zeros(5, np.float32), 0),
        ("nonfinite threshold", np.zeros(5, np.float32), float("nan")),
        ("boolean threshold", np.zeros(5, np.float32), True),
    ]
    for name, values, threshold in invalid:
        try:
            event_steps(values, threshold)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("Expected rejection: " + name)
    report = dict(schema="mac-float32-event-policy-constructed-review-v1", all_passed=True,
                  reference_arithmetic=EVENT_ARITHMETIC, legacy_commit=PIN, legacy_source_sha256=LEGACY_SHA,
                  registered_grid_points=138, legacy_equivalent_trace_comparisons=comparisons,
                  attainable_uint8_five_byte_sums_checked=sums_checked,
                  explicit_state_machine_cases=[r[0] for r in known],
                  non_byte_float_precision_preserved=True, exact_probability_one_supported=True,
                  terminal_clock_clamping_verified=True, rejected_cases=rejected,
                  source_sha256=sha(Path(__file__).with_name("float32_event_policy_mac_reference_v1.py").read_bytes()),
                  review_sha256=sha(Path(__file__).read_bytes()), model_invocations=0, media_reads=0,
                  thresholds_selected=False, original_evaluators_changed=False,
                  limits="Constructed arithmetic/FSM checks only; opt-in independent Mac reference is not an activated collector or recognition protocol. Quantized witness equivalence does not prove raw float quality or model equivalence. No model/development/acceptance/audio data consumed, no scoring or threshold selection.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
