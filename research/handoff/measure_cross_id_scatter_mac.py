"""Describe cross-conditioning-ID scatter from the saved original v46 arrays.

Use every endpoint common to all672IDs at each speed, for every phrase/arm.
Compute exact mean pair squared distance/cosine distance from sufficient sums;
do not select peaks, individual pairs, thresholds, a loss or a training weight.
These are already examined synthetic training IDs, not independent speakers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPORT = "46443b0a0bb835a9642d4be580974f9438ed527fba1e641b2132cbd2c88e4f8e"
PHRASES = ("hey_nyx", "hey_nick", "hey_next", "hey_mix")
ARMS = ("quiet", "noise25", "combined", "noise15", "rir")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pair_mean_squared(total, square_total, count):
    """Exact ordered-distinct-pair squared distance, without enumerating pairs."""
    return 2 * (count * square_total - np.sum(total * total, axis=-1)) / (count * (count - 1))


def pair_cosine_distance(unit_total, count):
    return (count * count - np.sum(unit_total * unit_total, axis=-1)) / (count * (count - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh diagnostic output required")
    # Independent small brute-force identity check, before real-data summaries.
    toy = np.array([[1., 2.], [-2., .5], [.25, 3.]])
    toy_unit = toy / np.sqrt(np.sum(toy * toy, axis=1))[:, None]
    distances = [np.sum((a - b) ** 2) for i, a in enumerate(toy)
                 for j, b in enumerate(toy) if i != j]
    cosines = [1 - a @ b for i, a in enumerate(toy_unit)
               for j, b in enumerate(toy_unit) if i != j]
    np.testing.assert_allclose(pair_mean_squared(toy.sum(0), np.sum(toy * toy), 3), np.mean(distances))
    np.testing.assert_allclose(pair_cosine_distance(toy_unit.sum(0), 3), np.mean(cosines))
    root = args.input.resolve()
    report_bytes = (root / "report.json").read_bytes()
    if sha(report_bytes) != REPORT:
        raise ValueError("Expected the completed original Mac v46 measurement")
    report = json.loads(report_bytes)
    groups_bytes = (root / "groups.json").read_bytes()
    if sha(groups_bytes) != report["groups_sha256"]:
        raise ValueError("Group identity differs")
    groups = json.loads(groups_bytes)
    all_results = []
    receipts = []
    for speed in (.9, 1., 1.1):
        rows = [r for r in groups if r["speed"] == speed]
        if len(rows) != 672 or len({r["voice"] for r in rows}) != 672:
            raise ValueError("Incomplete ID/speed grid")
        # Discover geometry only; no classifier values used to choose endpoints.
        shared = None
        for row in rows:
            with np.load(root / row["arrays"], allow_pickle=False) as data:
                ends = set(data["ends"].tolist())
            shared = ends if shared is None else shared & ends
        ends = sorted(shared)
        if not ends:
            raise ValueError("No common endpoint across all conditioning IDs")
        n, shape = len(rows), (4, 5, len(ends), 300)
        total = np.zeros(shape)
        total_unit = np.zeros(shape)
        square_total = np.zeros(shape[:-1])
        near_squared = np.zeros((3, 5, len(ends)))
        near_cosine = np.zeros_like(near_squared)
        norm_sum = np.zeros(shape[:-1])
        for row in rows:
            path = root / row["arrays"]
            if sha(path.read_bytes()) != row["arrays_sha256"]:
                raise ValueError("Saved classifier array identity differs")
            receipts.append((row["arrays"], row["arrays_sha256"]))
            with np.load(path, allow_pickle=False) as data:
                indices = np.searchsorted(data["ends"], ends)
                np.testing.assert_array_equal(data["ends"][indices], ends)
                values = data["embeddings"][:, :, indices].astype(np.float64)
            norm_squared = np.sum(values * values, axis=-1)
            norm = np.sqrt(norm_squared)
            if values.shape != shape or not np.isfinite(values).all() or np.any(norm == 0):
                raise ValueError("Invalid or undefined representation")
            unit = values / norm[..., None]
            total += values
            total_unit += unit
            square_total += norm_squared
            norm_sum += norm
            near_squared += np.sum((values[1:] - values[:1]) ** 2, axis=-1)
            near_cosine += 1 - np.sum(unit[1:] * unit[:1], axis=-1)
        within_squared = pair_mean_squared(total, square_total, n)
        within_cosine = pair_cosine_distance(total_unit, n)
        metrics = {}
        for p, phrase in enumerate(PHRASES):
            for a, arm in enumerate(ARMS):
                prefix = f"within_phrase/{phrase}/{arm}/"
                metrics[prefix + "distinct_id_mean_squared_distance"] = within_squared[p, a].tolist()
                metrics[prefix + "distinct_id_mean_cosine_distance"] = within_cosine[p, a].tolist()
                metrics[prefix + "mean_original_norm"] = (norm_sum[p, a] / n).tolist()
        for p, phrase in enumerate(PHRASES[1:]):
            # Off-diagonal target/near-word ID pairs. Subtract all matching-ID
            # pairs from the full Cartesian product; no sampled pair ordering.
            all_pair_squared = (square_total[0] + square_total[p + 1]) / n - 2 * np.sum(
                total[0] * total[p + 1], axis=-1) / (n * n)
            all_pair_cosine = 1 - np.sum(total_unit[0] * total_unit[p + 1], axis=-1) / (n * n)
            for a, arm in enumerate(ARMS):
                prefix = f"near_word/{phrase}/{arm}/"
                metrics[prefix + "same_id_mean_squared_distance"] = (near_squared[p, a] / n).tolist()
                metrics[prefix + "same_id_mean_cosine_distance"] = (near_cosine[p, a] / n).tolist()
                metrics[prefix + "distinct_id_mean_squared_distance"] = (
                    (n * all_pair_squared[a] - near_squared[p, a] / n) / (n - 1)).tolist()
                metrics[prefix + "distinct_id_mean_cosine_distance"] = (
                    (n * all_pair_cosine[a] - near_cosine[p, a] / n) / (n - 1)).tolist()
        if not all(np.isfinite(v).all() for v in metrics.values()):
            raise ValueError("Undefined summary")
        all_results.append(dict(speed=speed, conditioning_ids=n, ends=ends, metrics=metrics))
        print(json.dumps(dict(speed=speed, common_ends=len(ends), conditioning_ids=n)), flush=True)
    keys = sorted(all_results[0]["metrics"])
    result = dict(scope=__doc__, input_report_sha256=REPORT,
                  source_array_identity_receipt_sha256=sha(json.dumps(sorted(receipts)).encode()),
                  source_groups_authenticated=len(receipts), dimensions=300,
                  all_common_endpoints_included=True, brute_force_formula_selfcheck=True,
                  per_speed_endpoint_metrics=all_results,
                  mean_over_endpoints_then_three_speeds={
                      key: float(np.mean([np.mean(r["metrics"][key]) for r in all_results])) for key in keys},
                  script_sha256=sha(Path(__file__).read_bytes()),
                  limits="Exact descriptive pair averages on already examined synthetic training IDs. "
                         "Squared Euclidean distance is not mean Euclidean distance. No real-speaker, "
                         "causal, event-FAR, gradient, loss-effectiveness, export or P4 acceptance claim.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
