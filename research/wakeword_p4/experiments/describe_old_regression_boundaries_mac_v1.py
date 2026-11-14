"""Add timing descriptions to original saved old-regression event ledgers.

Never reads score arrays, invokes a model, computes new events, selects a
threshold, changes original reports, or accesses fresh development/acceptance.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np


PREFIX = 'research/wakeword_p4/'
PARENT_REF = 'f0449e3d45cb8b68f26bf9e7a3a54857ae978e87'
PARENT = PREFIX + 'results/4090-old-fixed-grid-regression-plan-v2.json'
PLAN = PREFIX + 'results/mac-old-regression-boundary-reader-plan-v1.json'
CHECK = PREFIX + 'results/mac-old-regression-boundary-reader-constructed-v1.json'
BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample',
              'generated_container_end_sample', 'full_rir_container_end_sample')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tracked(repo, ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=repo)


def stats(values):
    return dict(count=len(values), minimum=float(min(values)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(max(values))) if values else dict(
                    count=0, minimum=None, median=None, p95=None, maximum=None)


def describe(rows, target, segments):
    assert len(rows) == len(segments)
    counts = dict(positive_clips=0, any_event_clips=0, after_source_clips=0,
                  pre_source_only_clips=0, only_before_dry_prefix_end_clips=0,
                  only_before_rir_prefix_end_clips=0,
                  dry_complete_clips=0, rir_complete_clips=0)
    offsets = {key: dict(first_after_source=[], first_at_or_after_boundary=[])
               for key in BOUNDARIES}
    for row, steps in zip(rows, segments, strict=True):
        steps = np.asarray(steps)
        assert steps.dtype == np.int64 and steps.ndim == 1
        frames = max(0, (row['samples'] + 159) // 160 - 2)
        assert np.all(steps >= 5) and np.all(steps <= (frames + 2) // 3)
        assert np.all(np.diff(steps) > 0)
        if row['target_id'] != target:
            continue
        counts['positive_clips'] += 1
        assert type(row['lead_samples']) is int and 0 <= row['lead_samples'] <= row['samples']
        assert all(type(row[key]) is int and row[key] >= 0 for key in BOUNDARIES)
        # The original ledger uses one-based inference steps. Preserve its
        # final incomplete-hop clock clamp, including boundary equality.
        times = [min((3 * int(step) + 2) * 160, row['samples']) for step in steps]
        after_source = [time for time in times if time >= row['lead_samples']]
        dry = [time for time in times if time >= row[BOUNDARIES[0]]]
        rir = [time for time in times if time >= row[BOUNDARIES[1]]]
        counts['any_event_clips'] += int(bool(times))
        counts['after_source_clips'] += int(bool(after_source))
        counts['pre_source_only_clips'] += int(bool(times) and not after_source)
        counts['only_before_dry_prefix_end_clips'] += int(bool(after_source) and not dry)
        counts['only_before_rir_prefix_end_clips'] += int(bool(after_source) and not rir)
        counts['dry_complete_clips'] += int(bool(dry))
        counts['rir_complete_clips'] += int(bool(rir))
        for key in BOUNDARIES:
            if after_source:
                offsets[key]['first_after_source'].append((after_source[0] - row[key]) / 16)
            complete = [time for time in times if time >= row[key]]
            if complete:
                offsets[key]['first_at_or_after_boundary'].append((complete[0] - row[key]) / 16)
    return dict(counts=counts, boundaries={key: {
        label + '_signed_ms': stats(values) for label, values in groups.items()}
        for key, groups in offsets.items()})


def selfcheck(a):
    row = dict(samples=6000, target_id='hey_nyx', lead_samples=3200,
               generated_prefix_end_sample=3680, full_rir_prefix_end_sample=4160,
               generated_container_end_sample=4640, full_rir_container_end_sample=5120)
    segments = [np.array(v, np.int64) for v in [[5], [6], [7, 8, 9, 10], []]]
    result = describe([dict(row) for _ in segments], 'hey_nyx', segments)
    assert result['counts'] == dict(positive_clips=4, any_event_clips=3,
        after_source_clips=2, pre_source_only_clips=1,
        only_before_dry_prefix_end_clips=1, only_before_rir_prefix_end_clips=1,
        dry_complete_clips=1, rir_complete_clips=1)
    dry = result['boundaries'][BOUNDARIES[0]]
    observed = dry['first_after_source_signed_ms']
    assert {key: value for key, value in observed.items() if key != 'p95'} == dict(
        count=2, minimum=-30., median=-15., maximum=0.)
    assert abs(observed['p95'] - (-1.5)) < 2e-14
    assert all(result['boundaries'][key]['first_at_or_after_boundary_signed_ms'] ==
               dict(count=1, minimum=0., median=0., p95=0., maximum=0.) for key in BOUNDARIES)
    tail = dict(row, samples=5100, full_rir_container_end_sample=5100)
    assert describe([tail], 'hey_nyx', [np.array([10], np.int64)])[
        'boundaries'][BOUNDARIES[3]]['first_at_or_after_boundary_signed_ms']['maximum'] == 0.
    assert describe([dict(row, target_id='hey_willow')], 'hey_nyx', [segments[2]])['counts']['positive_clips'] == 0
    rejects = 0
    for bad in [[4], [0], [7, 7], [8, 7], [14]]:
        try:
            describe([row], 'hey_nyx', [np.array(bad, np.int64)])
        except AssertionError:
            rejects += 1
    assert rejects == 5
    report = dict(schema='mac-old-regression-boundary-reader-constructed-v1',
        script_sha256=sha(Path(__file__).read_bytes()), all_passed=True,
        explicit_count_vector_exact=True, signed_dry_offsets_exact=True,
        all_four_equality_boundaries_exact=True, final_clock_clamp_exact=True,
        other_target_excluded=True, malformed_step_rejections=rejects,
        limits='Constructed ledgers/metadata only; no actual regression events, scores, models, media or acceptance accessed.')
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


def plan(a):
    repo = a.repository_root.resolve()
    data = tracked(repo, PARENT_REF, PARENT)
    lock = json.loads(data)
    assert lock['role'] == 'reused-development-regression' and len(lock['models']) == 5
    files = [PREFIX + 'experiments/describe_old_regression_boundaries_mac_v1.py',
             PREFIX + 'experiments/old_regression_boundary_supplement_mac_v1.md', CHECK]
    check = json.loads((repo / CHECK).read_text())
    assert check['all_passed'] is True
    assert check['script_sha256'] == sha(Path(__file__).read_bytes())
    result = dict(schema='mac-old-regression-boundary-reader-plan-v1',
        parent_ref=PARENT_REF, parent_path=PARENT, parent_sha256=sha(data),
        file_sha256={name: sha((repo / name).read_bytes()) for name in files},
        boundaries=list(BOUNDARIES), models=lock['models'], expected_points=1242,
        role='reused-development-regression-descriptive-only',
        policy='After original collection/reduction/independent audit are published, describe their original full event ledgers at all existing138 points. Preserve original outcomes, thresholds, clocks and reports. Four generated-reference boundaries and source/early-only counts, no new score/event computation, inference, coefficient/threshold selection, promotion, fresh-development or acceptance access. Negative offsets do not prove an incorrect reference or a successful human wakeup.')
    with (repo / PLAN).open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=sha((repo / PLAN).read_bytes()), parent_sha256=sha(data))))


def run(a):
    repo = a.repository_root.resolve()
    plan_bytes = (repo / PLAN).read_bytes()
    assert tracked(repo, a.published_ref, PLAN) == plan_bytes
    frozen = json.loads(plan_bytes)
    for name, expected in frozen['file_sha256'].items():
        data = (repo / name).read_bytes()
        assert sha(data) == expected and tracked(repo, a.published_ref, name) == data
    parent_bytes = (repo / PARENT).read_bytes()
    assert sha(parent_bytes) == frozen['parent_sha256']
    assert parent_bytes == tracked(repo, PARENT_REF, PARENT)
    lock = json.loads(parent_bytes)
    assert lock['models'] == frozen['models'] and lock['role'] == 'reused-development-regression'
    assert importlib.metadata.version('numpy') == lock['versions']['numpy']
    result_ref = subprocess.check_output(['git', 'rev-parse', a.gpu_results_ref], cwd=repo, text=True).strip()
    originals = {}
    identities = {}
    for label, filename in [('collection', 'collection'), ('metrics', 'metrics'),
                            ('audit', 'independent-audit')]:
        name = PREFIX + 'results/4090-old-fixed-grid-regression-' + filename + '-v2.json'
        data = (repo / name).read_bytes()
        assert tracked(repo, result_ref, name) == data
        originals[label] = json.loads(data)
        identities[label] = sha(data)
    collection, metrics, audit = [originals[key] for key in ['collection', 'metrics', 'audit']]
    assert collection['plan_sha256'] == metrics['plan_sha256'] == sha(parent_bytes)
    assert metrics['collection_sha256'] == audit['collection_sha256'] == identities['collection']
    assert audit['metrics_sha256'] == identities['metrics'] and metrics['selection_authorized'] is False
    assert len(audit['summary']) == 9 and all(
        r['points'] == 138 and r['complete_event_ledgers_proved'] is True and
        r['all_metric_reductions_exact'] is True for r in audit['summary'])
    manifest_path = lock['synthetic_manifest']
    manifest_bytes = (repo / manifest_path).read_bytes()
    assert sha(manifest_bytes) == lock['files'][manifest_path]
    rows = json.loads(manifest_bytes)['records']
    assert len(rows) == 360 and len(collection['records']) == 1860
    receipts = {(r['model'], r['phase'], r['index']): r for r in collection['records']}
    assert len(receipts) == 1860
    grid = sorted(set([.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994] +
                      [.5 + index / 256 for index in range(128)]))
    expected = {(m['id'], kind, t) for m in lock['models'] for kind in
                (['uint8', 'float'] if m['weights'] is not None else ['uint8']) for t in grid}
    observed = set()
    models = {m['id']: m for m in lock['models']}
    outputs = []
    for point in metrics['points']:
        key = point['model'], point['kind'], point['threshold']
        assert key in expected and key not in observed
        observed.add(key)
        assert point['target'] == models[point['model']]['target']
        for index, row in enumerate(rows):
            receipt = receipts[point['model'], 'synthetic', index]
            assert receipt['source_id'] == row['source_id'] and receipt['samples'] == row['samples']
        ledger_path = repo / point['events_path']
        assert sha(ledger_path.read_bytes()) == point['events_sha256']
        with np.load(ledger_path, allow_pickle=False) as ledger:
            offsets, steps = ledger['synthetic_offsets'], ledger['synthetic_steps']
            assert offsets.dtype == steps.dtype == np.int64 and offsets.shape == (361,)
            assert offsets[0] == 0 and offsets[-1] == len(steps) and np.all(np.diff(offsets) >= 0)
            result = describe(rows, point['target'], [steps[offsets[j]:offsets[j + 1]] for j in range(360)])
        assert result['counts']['positive_clips'] == point['synthetic']['positive_clips'] == 36
        assert result['counts']['dry_complete_clips'] == point['synthetic']['complete_phrase_activated']
        assert result['counts']['rir_complete_clips'] / 36 == point['synthetic']['full_rir_prefix_recall']
        for boundary, label in [(BOUNDARIES[0], 'first_complete_hit_generated_reference_ms'),
                                (BOUNDARIES[1], 'first_rir_complete_hit_reference_ms')]:
            assert result['boundaries'][boundary]['first_at_or_after_boundary_signed_ms'] == point['synthetic'][label]
        outputs.append(dict(model=point['model'], kind=point['kind'], threshold=point['threshold'],
                            target=point['target'], original_events_sha256=point['events_sha256'], **result))
    assert observed == expected and len(outputs) == frozen['expected_points']
    report = dict(schema='mac-old-regression-boundary-supplement-v1',
        plan_sha256=sha(plan_bytes), gpu_results_ref=result_ref,
        original_report_sha256=identities, synthetic_manifest_sha256=sha(manifest_bytes),
        points=outputs, original_prefix_first_hit_statistics_exact=True,
        selection_authorized=False, promotion_authorized=False, limits=frozen['policy'])
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(points=len(outputs), report_sha256=sha(a.report.read_bytes()),
                          original_prefix_first_hit_statistics_exact=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['selfcheck', 'plan', 'run'])
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    parser.add_argument('--published-ref')
    parser.add_argument('--gpu-results-ref')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.action == 'run' and not (args.published_ref and args.gpu_results_ref and args.report):
        parser.error('run requires published-ref, gpu-results-ref and a new report path')
    if args.action == 'selfcheck' and not args.report:
        parser.error('selfcheck requires a new report path')
    {'selfcheck': selfcheck, 'plan': plan, 'run': run}[args.action](args)
