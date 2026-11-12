"""Independently recalculate the frozen Mac diagnostic from its saved arrays.

No model or feature inference. Do not import the measurement implementation:
check all geometry, file identities, head probabilities and 257 descriptive
metrics independently, including the equal-speed/equal-ID aggregation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Fresh audit output required")
    root = args.input.resolve()
    report_bytes = (root / "report.json").read_bytes()
    report = json.loads(report_bytes)
    raw_groups = (root / "groups.json").read_bytes()
    check(digest(raw_groups) == report["groups_sha256"], "Group file identity differs")
    groups = json.loads(raw_groups)
    raw_sources = (root / "factorized-original-report.json").read_bytes()
    check(digest(raw_sources) == report["source_report_sha256"], "Source report differs")
    sources = json.loads(raw_sources)["records"]
    head_file = (root / "classifier.npz").read_bytes()
    check(digest(head_file) == report["classifier_npz_sha256"], "Classifier identity differs")
    with np.load(root / "classifier.npz", allow_pickle=False) as head:
        weight = head["weight"].astype(np.float64)
        bias = float(head["bias"])
    head_length = np.sqrt(np.sum(weight * weight))
    check(weight.shape == (300,) and head_length > 0, "Unexpected classifier")
    phrases = ["hey_nyx", "hey_nick", "hey_next", "hey_mix"]
    arms = ["quiet", "noise25", "combined", "noise15", "rir"]
    check(report["phrase_order"] == phrases and report["arm_order"] == arms,
          "Axis order differs")
    voices = sorted(str(v) for v in np.random.default_rng(726).permutation(904)[:672])
    voices.sort(key=int)
    keys = sorted(report["summaries"])
    check(len(groups) == 2016 and len(keys) == 257, "Incomplete measurement census")
    group_means = np.empty((672, 3, 257), dtype=np.float64)
    seen, windows, maximum_probability_error = set(), 0, 0.0
    maximum_summary_error = 0.0

    def compare(actual, expected):
        nonlocal maximum_summary_error
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)
        maximum_summary_error = max(maximum_summary_error,
                                    float(np.max(np.abs(np.asarray(actual) - expected))))

    for i, group in enumerate(groups):
        check((group["voice"], group["speed"]) ==
              (voices[i // 3], (.9, 1., 1.1)[i % 3]), "Group order/cell differs")
        check(group["arrays"] == f"group-{i}.npz", "Unexpected array path")
        data = (root / group["arrays"]).read_bytes()
        check(digest(data) == group["arrays_sha256"], "Saved array identity differs")
        rows = [sources[index] for index in group["pack_indices"]]
        check(len(rows) == 20 and not (set(group["pack_indices"]) & seen),
              "Duplicate or missing source")
        seen.update(group["pack_indices"])
        check([(r["voice"], r["length_multiplier"], r["target"], r["arm"]) for r in rows] ==
              [(group["voice"], group["speed"], p, a) for p in phrases for a in arms],
              "Saved source axes differ")
        expected_ends = [e for e in range(150, 351) if e % 3 == 0
                         and all(r["feature_crop_end"] <= e < r["raw_shape"][0]
                                 for r in rows)]
        with np.load(root / group["arrays"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ends"], expected_ends)
            embedding = saved["embeddings"].astype(np.float64)
            probability = saved["probabilities"].astype(np.float64)
        n = len(expected_ends)
        check(n == group["windows"] and embedding.shape == (4, 5, n, 300)
              and probability.shape == (4, 5, n) and n > 0
              and np.isfinite(embedding).all() and np.isfinite(probability).all(),
              "Invalid array dimensions/values")
        windows += n
        logits = np.sum(embedding * weight, axis=-1) + bias
        recovered = np.exp(-np.logaddexp(0., -logits))
        np.testing.assert_allclose(recovered, probability, atol=2e-6, rtol=2e-5)
        maximum_probability_error = max(maximum_probability_error,
                                        float(np.max(np.abs(recovered - probability))))
        values = {}

        def paired(prefix, x, y):
            dx = y - x
            nx = np.sqrt(np.sum(x * x, axis=-1))
            ny = np.sqrt(np.sum(y * y, axis=-1))
            check(np.all(nx > 0) and np.all(ny > 0), "Unexpected undefined cosine")
            length = np.sqrt(np.sum(dx * dx, axis=-1))
            projection = np.sum(dx * weight, axis=-1)
            parallel = np.abs(projection) / head_length
            metrics = dict(l2=length, cosine_distance=1 - np.clip(
                np.sum((x / nx[:, None]) * (y / ny[:, None]), axis=-1), -1, 1),
                left_norm=nx, right_norm=ny, logit_delta=projection,
                parallel_l2=parallel,
                orthogonal_l2=np.sqrt(np.maximum(0., length ** 2 - parallel ** 2)))
            values.update({prefix + key: value for key, value in metrics.items()})

        for p, phrase in enumerate(phrases):
            for a, arm in enumerate(arms):
                prefix = f"output/{phrase}/{arm}/"
                values[prefix + "logit"] = logits[p, a]
                values[prefix + "probability"] = probability[p, a]
                if a:
                    paired(f"nuisance/{phrase}/{arm}/", embedding[p, 0], embedding[p, a])
                if p:
                    paired(f"near_word/{phrase}/{arm}/", embedding[0, a], embedding[p, a])
        check(sorted(values) == keys == sorted(group["measurements"]), "Metric set differs")
        for k, key in enumerate(keys):
            v, recorded = values[key], group["measurements"][key]
            check(recorded["count"] == recorded["defined"] == n
                  and recorded["undefined"] == 0 and np.isfinite(v).all(),
                  "Unexpected undefined window")
            compare(np.array([v.mean(), v.min(), v.max()]),
                    np.array([recorded["mean"], recorded["min"], recorded["max"]]))
            group_means[i // 3, i % 3, k] = v.mean()
        if (i + 1) % 336 == 0:
            print(json.dumps(dict(phase="audit_saved_arrays", groups=i + 1)), flush=True)
    check(seen == set(range(40320)) and windows == report["common_absolute_windows"]
          and windows * 20 == report["measured_classifier_inputs"], "Incomplete source/window census")
    id_means = group_means.mean(axis=1)
    for k, key in enumerate(keys):
        v, recorded = id_means[:, k], report["summaries"][key]
        check(recorded["count"] == recorded["defined"] == recorded["total_conditioning_ids"] == 672
              and all(recorded[field] == 0 for field in (
                  "undefined", "undefined_conditioning_ids", "undefined_groups",
                  "partially_defined_groups", "undefined_windows")), "Incomplete ID aggregation")
        compare(np.array([v.mean(), v.min(), v.max()]),
                np.array([recorded["mean"], recorded["min"], recorded["max"]]))
        compare(np.quantile(v, [.05, .25, .5, .75, .95]),
                np.array([recorded["quantiles"][q] for q in ("p05", "p25", "p50", "p75", "p95")]))
    result = dict(scope=__doc__, report_sha256=digest(report_bytes),
                  groups_sha256=digest(raw_groups), groups=len(groups),
                  complete_source_census=len(seen), common_absolute_windows=windows,
                  classifier_inputs=windows * 20, dimensions=300, metric_keys=257,
                  group_metric_sets_recalculated=len(groups) * 257,
                  equal_speed_equal_id_aggregate_sets_recalculated=257,
                  undefined_metric_values=0, all_array_sha256_match=True,
                  all_head_probabilities_reconstruct=True,
                  maximum_probability_error=maximum_probability_error,
                  maximum_summary_absolute_error=maximum_summary_error,
                  script_sha256=digest(Path(__file__).read_bytes()),
                  limits="Independent saved-array arithmetic and geometry; not another model inference, "
                         "full-domain export proof, event gate or generalization test.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
