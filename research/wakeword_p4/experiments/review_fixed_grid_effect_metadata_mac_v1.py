"""Recompute matched compilation effects from published old-case statistics.

No model/array inference is performed; this is not an admission calculation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess


def review(ref):
    ref = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    sources = {}
    def read(name):
        path = 'research/wakeword_p4/results/4090-hey-broad-' + name + '-v1.json'
        raw = subprocess.check_output(['git', 'show', ref + ':' + path])
        sources[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    ordinary = read('paired-qat-screen-result')
    fixed = read('fixed-grid-screen-result')
    independent = read('fixed-grid-screen-independent-audit')
    rounding = read('fixed-grid-input-rounding-result')
    maps = []
    for result in [ordinary, fixed, rounding]:
        mapping = {(c['target'], c['condition'], c['request_order']): c for c in result['cases']}
        assert len(result['cases']) == len(mapping) == 540
        maps.append(mapping)
    assert maps[0].keys() == maps[1].keys() == maps[2].keys()
    summary = defaultdict(lambda: dict(cases=0, ordinary_passed=0, fixed_passed=0,
        both_passed=0, ordinary_failed_fixed_passed=0, ordinary_passed_fixed_failed=0,
        both_failed=0, failed_requests=[], input_rounding_above_point1=0,
        remaining_p99_above_point1=0, max_input_rounding_p99=0.))
    for key in sorted(maps[0]):
        old, new, rounded = [m[key] for m in maps]
        recomputed = []
        for c in [old, new]:
            e = c['smoothed_abs_error']
            assert all(math.isfinite(v) for v in e.values())
            p = (c['repeat_exact'] and c['phase_max_error'] <= .0001 and
                 e['mean'] <= .02 and e['p99'] <= .1 and e['maximum'] <= .25)
            assert p == c['passed']
            recomputed.append(p)
        assert old['input_quantization'] == new['input_quantization']
        assert old['output_quantization'] == new['output_quantization']
        assert rounded['twice_reset_exact']
        for field, source_field in [('mean', 'mean_absolute'), ('p99', 'p99_absolute'), ('maximum', 'maximum_absolute')]:
            assert abs(new['smoothed_abs_error'][field] - rounded['total_original'][source_field]) < 1e-14
        row = summary[key[1] + '/' + key[0]]
        row['cases'] += 1
        op, fp = recomputed
        row['ordinary_passed'] += op
        row['fixed_passed'] += fp
        row['both_passed'] += op and fp
        row['ordinary_failed_fixed_passed'] += not op and fp
        row['ordinary_passed_fixed_failed'] += op and not fp
        row['both_failed'] += not op and not fp
        row['max_input_rounding_p99'] = max(row['max_input_rounding_p99'], rounded['input_rounding']['p99_absolute'])
        if not fp:
            row['failed_requests'].append(key[2])
            row['input_rounding_above_point1'] += rounded['input_rounding']['p99_absolute'] > .1
            row['remaining_p99_above_point1'] += rounded['remaining_quantization']['p99_absolute'] > .1
    for key, row in summary.items():
        expected = independent['summary'][key]
        assert row['cases'] == 135 and row['fixed_passed'] == expected['passed'] < 135
        assert row['failed_requests'] == expected['failed_requests']
        assert row['input_rounding_above_point1'] == 0
        assert row['remaining_p99_above_point1'] == 135 - row['fixed_passed']
    return dict(schema='mac-fixed-grid-compilation-effect-metadata-review-v1', gpu_ref=ref,
        source_sha256=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recomputed_summary=dict(summary), ordinary_failures=sum(135-r['ordinary_passed'] for r in summary.values()),
        fixed_failures=sum(135-r['fixed_passed'] for r in summary.values()),
        all_four_still_rejected=True, published_statistics_consistent=True,
        limits='Published whole540 per-case metadata only: recomputed unchanged gates, paired pass/fail transitions and rounded-input diagnostic statistics. No new actual inference, access to saved540 arrays/model bytes, unique-cause conclusion, recognition/FAR estimate, development selection or candidate/P4 admission. A different compiled graph/grid intervention changes the error pattern; numerical failures remain failures.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.gpu_ref)
    with args.output.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in ['ordinary_failures', 'fixed_failures', 'recomputed_summary']}))
