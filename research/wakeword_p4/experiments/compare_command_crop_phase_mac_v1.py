"""Recompute published full-coordinate phase reports without model inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def check_quantiles(values, reported):
    for q, expected in reported.items():
        assert math.isclose(quantile(values, float(q)), expected,
                            rel_tol=1e-12, abs_tol=1e-15), (q, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ['git', 'rev-parse', args.ref], cwd=args.repository, text=True).strip()
    files = {}

    def read(path):
        data = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=args.repository)
        files[path] = digest(data)
        return data

    prefix = 'research/wakeword_p4/results/4090-command-crop-phase'
    gpu, cpu = [json.loads(read(prefix + suffix + '-result-v1.json'))
                for suffix in ('', '-cpu')]
    locks = [json.loads(read(prefix + suffix + '-execution-v1.json'))
             for suffix in ('', '-cpu')]
    comparison = json.loads(read(prefix + '-backend-comparison-v1.json'))
    for suffix, report, lock in zip(('', '-cpu'), (gpu, cpu), locks, strict=True):
        assert report['execution_sha256'] == files[prefix + suffix + '-execution-v1.json']
        assert report['weights_unchanged'] and report['threshold_selected'] is False
        for name, expected in lock['files'].items():
            local = name if name.startswith('research/') else (
                'research/' + name.split('/research/', 1)[1] if '/research/' in name else None)
            if local is not None:
                assert digest(read(local)) == expected, local
    for name, expected in comparison['files'].items():
        assert files[name] == expected, name
    shared = locks[0]['files'].keys() & locks[1]['files'].keys()
    assert all(locks[0]['files'][name] == locks[1]['files'][name] for name in shared)
    assert cpu['runtime_device'] == 'CPU' and cpu['visible_gpus'] == []
    assert len(gpu['rows']) == len(cpu['rows']) == gpu['requests'] == cpu['requests'] == 12510
    gaps = {kind: [] for kind in ('nominal', 'phase')}
    for left, right in zip(gpu['rows'], cpu['rows'], strict=True):
        assert all(left[k] == right[k] for k in ('step', 'slot', 'new_order', 'new_end', 'stream'))
        assert left['new_end'] >= 225 and left['new_end'] % 3 == 0
        for report in (left, right):
            for key in ('nominal', 'phase', 'stream'):
                assert math.isfinite(report[key]) and 0 <= report[key] <= 1
            for key in gaps:
                assert abs(report[key] - report['stream']) == report[key + '_abs']
        for key in gaps:
            gaps[key].append(abs(left[key] - right[key]))
    for report in (gpu, cpu):
        for key in gaps:
            check_quantiles([row[key + '_abs'] for row in report['rows']], report[key + '_abs'])
    for key, values in gaps.items():
        check_quantiles(values, comparison['gpu_minus_cpu_abs'][key])
    errors = {key: max(row[key + '_abs'] for row in cpu['rows']) for key in gaps}
    disagreements = sum(row['nominal'] <= .5 and row['stream'] >= .9 for row in cpu['rows'])
    assert comparison['cpu_phase_within_original_1e_4'] == (errors['phase'] <= 1e-4)
    assert comparison['cpu_nominal_low_stream_high'] == disagreements == 9
    result = dict(schema='mac-published-command-crop-phase-comparison-v1',
        gpu_commit=commit, source_sha256=digest(Path(__file__).read_bytes()), files=files,
        all_coordinates_recomputed=12510, coordinate_and_saved_stream_equal=True,
        all_published_quantiles_recomputed=True,
        shared_execution_lock_bindings_equal=len(shared), cpu_nominal_max_abs=errors['nominal'],
        cpu_phase_max_abs=errors['phase'], cpu_phase_within_original_1e_4=True,
        cpu_nominal_low_stream_high=disagreements,
        gpu_cpu_max_abs={key: max(values) for key, values in gaps.items()},
        model_invocations=0, threshold_selected=False, candidate_qualified=False,
        limits='Independent arithmetic and committed-file/lock binding only. Private weights, '
               'features, saved original traces and actual runtime devices were not replayed '
               'or independently inspected on this Mac. Fixed-final endpoint differences '
               'do not prove training-time loss, causal recognition failure, training benefit, '
               'other-phase parity, FAR, INT8 or full-panel acceptance.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))


if __name__ == '__main__':
    main()
