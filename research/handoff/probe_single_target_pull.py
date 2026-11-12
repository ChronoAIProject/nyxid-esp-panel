"""Numerical counterexample for copying MACR's target-only pull to one keyword.

Implements the printed equation (1), not the complete MACR loss or a wake model.
No data, optimizer, gradients to model weights, inference or device is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def pull(vectors, labels, targets, temperature=.1):
    unit = []
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            raise ValueError("Nonzero embeddings required")
        unit.append([x / norm for x in vector])
    loss = 0.
    for i, (vector, label) in enumerate(zip(unit, labels)):
        positives = [j for j in range(len(unit)) if j != i and labels[j] == label]
        if label not in targets or not positives:
            continue
        denominator = [j for j in range(len(unit)) if j != i and labels[j] in targets]
        logits = {j: sum(x * y for x, y in zip(vector, unit[j])) / temperature
                  for j in denominator}
        maximum = max(logits.values())
        logsumexp = maximum + math.log(sum(math.exp(x - maximum) for x in logits.values()))
        loss += sum(logsumexp - logits[j] for j in positives) / len(positives)
    return loss


def pair(angle):
    return [[1., 0.], [math.cos(angle), math.sin(angle)], [0., -1.]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required")
    examples = []
    labels, targets = ["wake", "wake", "unknown"], {"wake"}
    for degrees in (0, 60, 90, 180):
        angle = math.radians(degrees)
        value = pull(pair(angle), labels, targets)
        delta = 1e-6
        derivative = (pull(pair(angle + delta), labels, targets)
                      - pull(pair(angle - delta), labels, targets)) / (2 * delta)
        if value != 0. or derivative != 0.:
            raise ValueError("Singleton-denominator analytic identity did not reproduce")
        examples.append(dict(angle_degrees=degrees, cosine=math.cos(angle),
                             printed_pull_loss=value, angle_derivative=derivative))
    collapsed = [[1., 0.]] * 3
    simplex = [[math.cos(x), math.sin(x)] for x in (0., 2 * math.pi / 3, 4 * math.pi / 3)]
    c, s = [pull(vectors, ["wake"] * 3, targets) for vectors in (collapsed, simplex)]
    expected = 3 * math.log(2)
    if max(abs(c - expected), abs(s - expected)) > 1e-12:
        raise ValueError("Equal-pair-logit analytic identity did not reproduce")
    result = dict(scope=__doc__,
                  source="https://www.isca-archive.org/interspeech_2026/zhang26ha_interspeech.pdf",
                  source_equation=1, temperature=.1, single_keyword_pair_examples=examples,
                  three_target_examples=dict(collapsed=c, separated_120_degree_simplex=s,
                                             analytic_value=expected),
                  all_analytic_identities_reproduced=True,
                  implication="With one target keyword and two target examples, equation1 "
                              "is identically zero regardless of their angle. With three "
                              "target examples, equal pairwise logits attain the same "
                              "pull loss for collapsed and separated regular-simplex "
                              "embeddings. Copying this component alone therefore does "
                              "not guarantee target compactness in the single-keyword case.",
                  limits="Not a criticism or reproduction of the paper's multi-keyword "
                         "experiment. The separate margin push and CE may still change "
                         "embeddings. No full-objective failure, model/gradient failure, "
                         "new recipe, data quality, wake performance or parity is claimed.",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
