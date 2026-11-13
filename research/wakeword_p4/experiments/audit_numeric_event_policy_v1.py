"""Audit analytic UINT8 boundaries against the compiled original C++ event core.

No model inference, new speech, playback, recording or board access. Optional
saved-v30 replay compares only the original frozen threshold on consumed data.
"""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--gpu-evidence-root', type=Path)
    parser.add_argument('--runtime-source', type=Path,
                        default=Path('research/panel_wake_runtime/panel_wake_runtime.cpp'))
    parser.add_argument('--evaluator-source', type=Path,
                        help='Exact archived 21fa349 evaluator, otherwise read from git')
    args = parser.parse_args()
    assert not args.output.exists()
    original = (args.evaluator_source.read_bytes() if args.evaluator_source else
        subprocess.check_output(['git', 'show', '21fa349:research/wakeword_p4/training/evaluate.py']))
    assert hashlib.sha256(original).hexdigest() == 'a7591fcadf00c1393887f9df99f201dced6d42a2111032f3803d842dd0086256'
    parsed = ast.parse(original)
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ('smooth', 'event_indices')]
    namespace = {'np': np, 'SMOOTH': 5}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original functions>', 'exec'), namespace)
    smooth, indices = namespace['smooth'], namespace['event_indices']
    grid = sorted(set([.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994]
                      + [.5 + i / 256 for i in range(128)]))
    assert len(grid) == 138
    rng = np.random.default_rng(202611131504)
    predicate_cases, window_count = [], 0
    for total in sorted({round(t * 1280) for t in grid} | {round(t * 640) for t in grid}):
        values, left = [], total
        for _ in range(5):
            value = min(255, left)
            values.append(value)
            left -= value
        windows = sorted(set(itertools.permutations(values)))
        candidates = rng.integers(0, 256, (12000, 4), dtype=np.int16)
        last = total - candidates.sum(axis=1)
        candidates = candidates[(last >= 0) & (last <= 255)]
        windows += [tuple(row.tolist() + [int(total - row.sum())]) for row in candidates[:100]]
        for window in windows:
            window_count += 1
            mean = float(smooth(np.asarray(window, np.float32) / 256)[0])
            device = sum(window) / 256 / 5
            thresholds = np.asarray(grid)
            device_thresholds = thresholds.astype(np.float32).astype(np.float64)
            different = ((mean >= thresholds) != (device >= device_thresholds)) | (
                (mean < thresholds / 2) != (device < device_thresholds / 2))
            for pos in np.flatnonzero(different):
                threshold = grid[pos]
                predicate_cases.append(dict(window=list(window), threshold=threshold,
                    host_mean=mean, device_mean=device, device_threshold=float(device_thresholds[pos]),
                    trigger_differs=bool((mean >= threshold) != (device >= device_thresholds[pos])),
                    rearm_differs=bool((mean < threshold / 2) != (device < device_thresholds[pos] / 2))))
    source = args.runtime_source
    assert sha(source) == '191321564faa65e30776f8869462c689794f2d1fe9f80c3d279c1d032dba18c1'
    runtime = source.read_text()
    core = runtime.split('            ++steps;\n', 1)[1].split('\n        }\n        return triggered?1:0;', 1)[0]
    core = '++steps;\n' + core
    assert core.count('PANEL_WAKE_THRESHOLD') == 2
    harness = '''#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#define ESP_LOGI(...) ((void)0)
#define PANEL_WAKE_THRESHOLD threshold
template<typename T> std::vector<int> run(const std::vector<int>& raw,T threshold) {
 constexpr int smooth_steps=5,refractory_steps=67;
 float history[5]{}; unsigned smooth_at=0,smooth_count=0,pcm_peak=0;
 int64_t steps=0,last=-1000; bool armed=true; double maximum_average=0;
 std::vector<int> found;
 for(int byte:raw) { float score=byte/256.f; bool triggered=false;
''' + core + '''
  if(triggered)found.push_back(int(steps));
 } return found;
}
void print(const std::vector<int>& v) { std::cout << "["; for(size_t i=0;i<v.size();++i) {if(i)std::cout<<",";std::cout<<v[i];} std::cout<<"]"; }
int main() {double t; int n; while(std::cin>>t>>n) {std::vector<int> raw(n);for(int& b:raw)std::cin>>b;
 std::cout<<"{\\"float\\":";print(run(raw,float(t)));std::cout<<",\\"double\\":";print(run(raw,t));std::cout<<"}\\n";}}
'''
    args.output.mkdir(parents=True)
    (args.output / 'event-core.cpp').write_text(harness)
    subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off', str(args.output / 'event-core.cpp'),
                    '-o', str(args.output / 'event-core')], check=True, capture_output=True)
    streams = []
    for case in predicate_cases:
        if case['trigger_differs']:
            raw = [0] * 10 + case['window'] + [0] * 10
            streams.append(dict(**case, kind='single-trigger', raw=raw))
        if case['rearm_differs']:
            raw = [255] * 80 + case['window'] + [255] * 10
            streams.append(dict(**case, kind='rearm-after-refractory', raw=raw))
    stdin = ''.join(f"{r['threshold']:.17g} {len(r['raw'])} " + ' '.join(map(str, r['raw'])) + '\n' for r in streams)
    completed = subprocess.run([str(args.output / 'event-core')], input=stdin, text=True, capture_output=True, check=True)
    compiled = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(compiled) == len(streams)
    failures, canonical_disagreements = [], []
    for row, device in zip(streams, compiled, strict=True):
        raw = np.asarray(row['raw'], np.float32) / 256
        means = smooth(raw)
        host = [i + 5 for i in indices(means, row['threshold'])]
        integers = np.convolve(np.asarray(row['raw'], np.int32), np.ones(5, np.int32), 'valid')
        canonical = integers.astype(np.float64) / 1280
        np.testing.assert_allclose(means, canonical, rtol=0, atol=4e-16)
        exact = [i + 5 for i in indices(canonical, row['threshold'])]
        if exact != device['double']:
            canonical_disagreements.append(dict(row=row, host=exact, device=device['double']))
        if host != device['float']:
            failures.append(dict(threshold=row['threshold'], kind=row['kind'], window=row['window'],
                host_event_steps=host, compiled_current_event_steps=device['float'],
                compiled_double_threshold_event_steps=device['double'], canonical_host_event_steps=exact))
    assert not canonical_disagreements
    v30 = [r for r in failures if r['threshold'] == .91796875]
    regression = []
    if args.baseline:
        assert sha(args.baseline / 'scores.npz') == '61e2fbfb31fdec528d9f572b9302c6d75267a49be524689f20ba2038dcf87711'
        with np.load(args.baseline / 'scores.npz', allow_pickle=False) as traces:
            assert len(traces.files) == 126
            for i in range(126):
                raw = traces[f'output_{i}']
                before = smooth(raw.astype(np.float32) / 256)
                after = np.convolve(raw.astype(np.int32), np.ones(5, np.int32), 'valid') / 1280
                regression.append(dict(cohort='original126', index=i,
                    old=indices(before, .91796875), canonical=indices(after, .91796875)))
    if args.gpu_evidence_root:
        for label, count in (('joe', 360), ('background', 42)):
            root = args.gpu_evidence_root / ('mac-' + label + '-score-receipt-v1/extracted/replay')
            report = json.loads((root / 'report.json').read_text())
            assert len(report['records']) == count
            for i, record in enumerate(report['records']):
                path = root / f'{i}-v30-scores.npy'
                assert sha(path) == record['v30']['score_sha256']
                before = np.load(path, allow_pickle=False)
                sums = np.rint(before * 1280)
                assert np.max(np.abs(before * 1280 - sums)) <= 1e-10
                after = sums / 1280
                regression.append(dict(cohort=label, index=i,
                    old=indices(before, .91796875), canonical=indices(after, .91796875)))
    changed = [r for r in regression if r['old'] != r['canonical']]
    (args.output / 'cases.json').write_text(json.dumps(dict(streams=streams, compiled=compiled), indent=2) + '\n')
    (args.output / 'regression.json').write_text(json.dumps(regression, indent=2) + '\n')
    report = dict(schema='mac-analytic-uint8-event-policy-boundary-audit-v1', source_sha256=sha(Path(__file__)),
        numpy=np.__version__, original_evaluator_commit='21fa349',
        evaluator_sha256=hashlib.sha256(original).hexdigest(), runtime_source_sha256=sha(source),
        extracted_event_core_sha256=hashlib.sha256(core.encode()).hexdigest(),
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
        compiler_flags=['-std=c++17', '-O2', '-ffp-contract=off'],
        grid_points=138, analytic_windows=window_count, predicate_differences=len(predicate_cases),
        constructed_streams=len(streams), compiled_original_event_disagreements=len(failures),
        disagreement_kinds=dict(Counter(r['kind'] for r in failures)),
        original_v30_boundary_examples=v30, other_examples=failures[:5],
        canonical_integer_mean_and_double_threshold_compiled_disagreements=0,
        consumed_frozen_v30_regression_clips=len(regression), changed_frozen_v30_event_streams=changed,
        cases_sha256=sha(args.output / 'cases.json'), regression_sha256=sha(args.output / 'regression.json'),
        model_inference=False, new_speech=False, training=False, firmware_changed=False, accepted=False,
        limits='Analytic score-byte counterexamples, not model-produced failure frequency or recognition '
            'quality. Compiled extracted existing C++ event core on Mac, not P4 execution. Canonical '
            'arithmetic is a proposed future policy clarification, not an applied calibration change; '
            'existing frozen reports/pins remain untouched. Saved-v30 replay uses only its frozen '
            'threshold, no threshold selection. Does not explain the dominant near-zero float misses.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
