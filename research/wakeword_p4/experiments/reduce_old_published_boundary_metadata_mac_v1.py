"""Describe published historical metadata without substituting for event intake.

No models, audio, score arrays or private event ledgers are opened. All original
fixed points remain in the reduction; no operating point is selected.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = 'research/wakeword_p4/results/'
RESULT_REF = 'ca79386ea4082036bae9875a2cfdccb8394f8b6d'
INPUTS = {
    'metrics': ('4090-old-fixed-grid-regression-metrics-v2.json',
                'e21dd80c39ec80e51bd411991d677d60acdf67a2d87fafa45f686bfe225d6a13'),
    'boundary': ('4090-old-fixed-grid-regression-boundary-supplement-v1.json',
                 'c2d713cc1f5ae987571d714c873097bbd939ff158c28df471c287c0950e0df81'),
}
BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample',
              'generated_container_end_sample', 'full_rir_container_end_sample')


def reduce(repo):
    originals = {}
    for label, (name, expected) in INPUTS.items():
        data = subprocess.check_output(
            ['git', 'show', RESULT_REF + ':' + PREFIX + name], cwd=repo)
        assert hashlib.sha256(data).hexdigest() == expected
        originals[label] = json.loads(data)
    metrics = originals['metrics']
    supplement = originals['boundary']
    assert metrics['selection_authorized'] is supplement['selection_authorized'] is False
    assert supplement['model_invocations'] == 0
    assert len(metrics['points']) == len(supplement['points']) == 1242
    by_key = {(p['model'], p['kind'], p['threshold']): p for p in metrics['points']}
    assert len(by_key) == 1242
    observed = set()
    groups = defaultdict(list)
    for point in supplement['points']:
        key = point['model'], point['kind'], point['threshold']
        assert key not in observed
        observed.add(key)
        original = by_key[key]
        assert point['events_sha256'] == original['events_sha256']
        rows = point['positive_rows']
        assert len(rows) == original['synthetic']['positive_clips'] == 36
        assert len({r['index'] for r in rows}) == 36
        assert original['synthetic']['pre_source_positive_events'] == 0
        for row in rows:
            assert type(row['event_count']) is int and row['event_count'] >= 0
            for boundary in BOUNDARIES:
                cell = row['boundaries'][boundary]
                assert (cell['first_event_signed_ms'] is None) == (row['event_count'] == 0)
                after = cell['first_at_or_after_ms']
                assert after is None or (row['event_count'] > 0 and after >= 0)
        any_count = sum(row['event_count'] > 0 for row in rows)
        dry = sum(row['boundaries'][BOUNDARIES[0]]['first_at_or_after_ms'] is not None
                  for row in rows)
        assert dry == original['synthetic']['complete_phrase_activated']
        assert 0 <= dry <= any_count <= 36
        counts = dict(any_event=any_count, complete_dry=dry,
                      early_only=any_count - dry, no_event=36 - any_count)
        for boundary in BOUNDARIES[1:]:
            counts[boundary + '_complete'] = sum(
                row['boundaries'][boundary]['first_at_or_after_ms'] is not None
                for row in rows)
        groups[key[:2]].append(counts)
    assert observed == set(by_key) and len(groups) == 9
    summary = []
    for (model, kind), points in sorted(groups.items()):
        assert len(points) == 138
        summary.append(dict(model=model, kind=kind, fixed_points=len(points),
                            count_ranges={key: [min(p[key] for p in points),
                                                max(p[key] for p in points)]
                                          for key in points[0]}))
    return dict(
        schema='mac-old-published-boundary-metadata-reduction-v1',
        source_ref=RESULT_REF,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_metrics_sha256=INPUTS['metrics'][1],
        original_gpu_boundary_supplement_sha256=INPUTS['boundary'][1],
        summary=summary, all1242_prefix_counts_match_original_metrics=True,
        selection_authorized=False, promotion_authorized=False,
        limits='Independent reduction of published GPU positive-row metadata only. '
               'Not independent event-ledger verification, model inference, score '
               'recomputation, new recognition scoring or candidate admission. All '
               '138 historical points retained without choosing a point. Generated '
               'boundaries are not phonetic truth. The original NPZ intake and frozen '
               '9162bdb four-boundary reader remain separately required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = reduce(args.repository_root.resolve())
    with args.report.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(points=1242, paths=len(report['summary']),
                          original_prefix_counts_exact=True,
                          report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest())))
