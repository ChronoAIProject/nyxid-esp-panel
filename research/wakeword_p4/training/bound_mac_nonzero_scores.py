"""Bound possible target hits by post-prime nonzero score support.

A positive threshold cannot trigger in a source whose complete post-prime
five-score sequence is zero. This necessary bound needs no zero-prime or
monotone-latch assumption and does not select or evaluate new thresholds.
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
    for name in ('artifacts', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    plan = json.loads(args.plan.read_bytes())
    result_bytes = (args.artifacts/'result.json').read_bytes()
    score_bytes = (args.artifacts/'scores.npz').read_bytes()
    assert sha(result_bytes) == plan['full_result_sha256']
    assert sha(score_bytes) == plan['scores_sha256']
    result = json.loads(result_bytes)
    assert result['accepted'] is False and result['sources'] == plan['sources'] == 350
    saved = np.load(args.artifacts/'scores.npz', allow_pickle=False)
    assert set(saved.files) == {f'output_{i}' for i in range(350)}
    rows = []
    for index, row in enumerate(result['source_details']):
        q = saved[f'output_{index}']
        assert q.dtype == np.uint8 and q.ndim == 1 and len(q) == row['output_count']
        assert sha(q.tobytes()) == row['output_sha256']
        assert row['prime_samples'] == plan['prime_samples'] == 56160
        cumulative = np.pad(np.cumsum(q, dtype=np.int64), (1, 0))
        sums = cumulative[5:] - cumulative[:-5]
        alternate = np.array([sum(int(v) for v in q[start:start+5])
                              for start in range(len(q)-4)], dtype=np.int64)
        np.testing.assert_array_equal(sums, alternate)
        # Score step 1 returns at PCM block 2; the first complete five-score
        # window returns at block 6. Preserve every prime and padded-tail step.
        ends = np.arange(6, len(q)+2)*480 - row['prime_samples']
        assert np.any(ends <= 0) and np.any(ends > 0)
        rows.append({k: row[k] for k in ('path', 'target', 'voice', 'variant', 'text', 'sha256')} | {
            'prime_max_five_score_sum': int(sums[ends <= 0].max()),
            'post_prime_max_five_score_sum': int(sums[ends > 0].max())})
    positives = [row for row in rows if row['target'] == 'hey_nyx']
    assert len(positives) == 50
    zeros = [row for row in positives if row['post_prime_max_five_score_sum'] == 0]
    cap = len(positives)-len(zeros)
    output = dict(scope=__doc__, accepted=False, sources=350,
                  full_result_sha256=sha(result_bytes), scores_sha256=sha(score_bytes),
                  plan_sha256=sha(args.plan.read_bytes()), source_sha256=sha(Path(__file__).read_bytes()),
                  all_output_hashes_verified=True, prefix_sum_and_direct_windows_exact=True,
                  prime_max_five_score_sum=max(row['prime_max_five_score_sum'] for row in rows),
                  denominator=1280, target_sources=50, all_zero_target_sources=len(zeros),
                  optimistic_target_hit_upper_bound_at_any_positive_threshold=cap,
                  descriptive_hiesp_reference_hits=plan['descriptive_hiesp_reference_hits'],
                  threshold_only_cannot_reach_reference=cap < plan['descriptive_hiesp_reference_hits'],
                  zero_target_witnesses=zeros,
                  by_voice={voice: dict(targets=len(group := [row for row in positives if row['voice'] == voice]),
                             all_zero=sum(row['post_prime_max_five_score_sum'] == 0 for row in group))
                            for voice in sorted({row['voice'] for row in positives})},
                  proof='Every event at any positive threshold requires a positive post-prime five-score sum. An all-zero source has no eligible score, independent of prime events or latch/refractory state. Nonzero support is an optimistic upper bound, not a claim that each supported source can trigger.',
                  zero_prime_certificate_applicable=all(row['prime_max_five_score_sum'] == 0 for row in rows),
                  limits=plan['limits'])
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({key: output[key] for key in ('prime_max_five_score_sum', 'all_zero_target_sources',
        'optimistic_target_hit_upper_bound_at_any_positive_threshold', 'threshold_only_cannot_reach_reference')}))


if __name__ == '__main__':
    main()
