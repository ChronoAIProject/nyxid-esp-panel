"""Pure geometry and descriptive metrics for the frozen balanced float probe.

No model, audio device or dataset is opened by this module. Undefined cosine
and relative distances remain explicitly undefined, including zero vectors.
"""
from collections import defaultdict
import itertools

import numpy as np


PHRASES = ("hey_nyx", "hey_nick", "hey_next", "hey_mix")
ARMS = ("quiet", "noise25", "combined", "noise15", "rir")
SPEEDS = (.9, 1., 1.1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_ends(rows):
    """All shared absolute stride-three ends for the known phase-one graph."""
    require(bool(rows), "Empty geometry group")
    low = max(row["feature_crop_end"] for row in rows)
    high = min(350, min(row["raw_shape"][0] - 1 for row in rows))
    # Phase-one full windows are [e-149:e+1]; e+1 must exist. Their effective
    # convolution history ends at e, because the last frame is unused.
    ends = np.arange((low + 2) // 3 * 3, high + 1, 3, dtype=np.int64)
    require(len(ends) > 0 and low >= 150, "No common complete-context window")
    return ends


def grouped_records(records, voices):
    """Require every declared ID/speed/phrase/arm cell exactly once."""
    cells = {}
    voices = tuple(voices)
    require(len(set(voices)) == len(voices), "Duplicate declared conditioning ID")
    for row in records:
        key = (row["voice"], row["length_multiplier"], row["target"], row["arm"])
        require(key not in cells, "Duplicate factorial cell")
        require(row["split"] == "training" and row["mechanical_pass"]
                and row["entire_signal_fits"] and len(row["raw_shape"]) == 2
                and row["raw_shape"][1] == 40, "Unexpected feature role or shape")
        cells[key] = row
    expected = set(itertools.product(voices, SPEEDS, PHRASES, ARMS))
    require(set(cells) == expected, "Incomplete or extra factorial cell")
    groups = []
    for voice in sorted(voices, key=int):
        for speed in SPEEDS:
            rows = [cells[voice, speed, phrase, arm]
                    for phrase in PHRASES for arm in ARMS]
            groups.append((voice, speed, rows, common_ends(rows)))
    return groups


def paired_metrics(left, right, classifier):
    """Per-window distances; classifier projection uses its actual weights."""
    left, right, classifier = [np.asarray(x, dtype=np.float64)
                               for x in (left, right, classifier)]
    require(left.ndim == right.ndim == 2 and left.shape == right.shape
            and left.shape[0] > 0 and classifier.shape == (left.shape[1],)
            and all(np.isfinite(x).all() for x in (left, right, classifier)),
            "Invalid paired representation")
    delta = right - left
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    l2 = np.linalg.norm(delta, axis=1)
    dot = np.einsum("ij,ij->i", left, right)
    product = left_norm * right_norm
    cosine = np.full(len(left), np.nan)
    valid = product > 0
    cosine[valid] = 1 - np.clip(dot[valid] / product[valid], -1, 1)
    head_norm = float(np.linalg.norm(classifier))
    logit_delta = delta @ classifier
    parallel = np.full(len(left), np.nan)
    orthogonal = np.full(len(left), np.nan)
    if head_norm > 0:
        parallel = np.abs(logit_delta) / head_norm
        orthogonal = np.sqrt(np.maximum(0, l2 ** 2 - parallel ** 2))
    return dict(l2=l2, cosine_distance=cosine, left_norm=left_norm,
                right_norm=right_norm, logit_delta=logit_delta,
                parallel_l2=parallel, orthogonal_l2=orthogonal)


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    known = values[finite]
    return dict(count=int(values.size), defined=int(finite.sum()),
                undefined=int(values.size - finite.sum()),
                mean=float(known.mean()) if len(known) else None,
                min=float(known.min()) if len(known) else None,
                max=float(known.max()) if len(known) else None)


def group_measurements(embeddings, probabilities, classifier, bias):
    """Equal-time comparisons without pooling the five temporal positions."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(embeddings.ndim == 4 and embeddings.shape[:2] == (4, 5)
            and embeddings.shape[2] > 0 and embeddings.shape[3] == 300
            and probabilities.shape == embeddings.shape[:3]
            and np.isfinite(embeddings).all() and np.isfinite(probabilities).all()
            and np.all((probabilities >= 0) & (probabilities <= 1)),
            "Unexpected complete classifier-input geometry")
    classifier = np.asarray(classifier, dtype=np.float64)
    logits = embeddings @ classifier + float(bias)
    reconstructed = np.empty_like(logits)
    positive = logits >= 0
    reconstructed[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_negative = np.exp(logits[~positive])
    reconstructed[~positive] = exp_negative / (1 + exp_negative)
    require(np.allclose(reconstructed, probabilities, atol=2e-6, rtol=2e-5),
            "Classifier projection does not reproduce actual probability")
    result = {}
    for p, phrase in enumerate(PHRASES):
        for a, arm in enumerate(ARMS):
            prefix = "output/" + phrase + "/" + arm + "/"
            result[prefix + "logit"] = finite_summary(logits[p, a])
            result[prefix + "probability"] = finite_summary(probabilities[p, a])
            if a:
                metrics = paired_metrics(embeddings[p, 0], embeddings[p, a], classifier)
                for name, values in metrics.items():
                    result["nuisance/" + phrase + "/" + arm + "/" + name] = finite_summary(values)
    for p, phrase in enumerate(PHRASES[1:], start=1):
        for a, arm in enumerate(ARMS):
            metrics = paired_metrics(embeddings[0, a], embeddings[p, a], classifier)
            for name, values in metrics.items():
                result["near_word/" + phrase + "/" + arm + "/" + name] = finite_summary(values)
    return result


def across_ids(rows):
    """Average within each speed/window group, then within each conditioning ID.

    Undefined comparisons stay counted. No confidence interval or independence
    claim is attached to synthetic conditioning IDs or their speed variants.
    """
    require(bool(rows), "No group summaries")
    keys = set(rows[0]["measurements"])
    by_id = defaultdict(list)
    for row in rows:
        require(set(row["measurements"]) == keys, "Inconsistent measurement set")
        by_id[row["voice"]].append(row)
    result = {}
    for key in sorted(keys):
        id_means = []
        undefined_windows = 0
        undefined_groups = 0
        partial_groups = 0
        for grouped in by_id.values():
            require(sorted(r["speed"] for r in grouped) == list(SPEEDS),
                    "Incomplete speed census for conditioning ID")
            means = []
            for row in grouped:
                item = row["measurements"][key]
                undefined_windows += item["undefined"]
                partial_groups += bool(item["undefined"] and item["defined"])
                if item["mean"] is None:
                    undefined_groups += 1
                else:
                    means.append(item["mean"])
            # Do not turn two defined speeds into a complete three-speed ID.
            if len(means) == 3:
                id_means.append(float(np.mean(means)))
        stats = finite_summary(id_means)
        stats.update(total_conditioning_ids=len(by_id),
                     undefined_conditioning_ids=len(by_id) - len(id_means),
                     undefined_groups=undefined_groups, partially_defined_groups=partial_groups,
                     undefined_windows=undefined_windows)
        if id_means:
            stats["quantiles"] = dict(zip(("p05", "p25", "p50", "p75", "p95"),
                                          np.quantile(id_means, [.05, .25, .5, .75, .95]).tolist()))
        result[key] = stats
    return result
