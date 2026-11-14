"""Recompute the two-arm phase experiment from every published point/group.

No model inference, audio access, threshold selection or private trace oracle.
The old four-arm comparer remains unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

BASE = 'research/wakeword_p4/results/'
DIMENSIONS = ('carrier_kind', 'literal', 'voice', 'source_id', 'length_scale', 'variant')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def span(values):
    return [min(values), max(values)] if values else [None, None]


def poisson_upper95(count, hours):
    """Invert P(Poisson(mu)<=count)=.05 without the GPU's chi-square library."""
    assert type(count) is int and 0 <= count < 100 and hours > 0
    def cdf(mu):
        term = math.exp(-mu)
        terms = [term]
        for k in range(1, count + 1):
            term *= mu / k
            terms.append(term)
        return math.fsum(terms)
    low, high = 0., max(8., 2. * (count + 1))
    while cdf(high) > .05:
        high *= 2
    for _ in range(100):
        mid = (low + high) / 2
        if cdf(mid) > .05:
            low = mid
        else:
            high = mid
    return ((low + high) / 2) / hours


def metrics(point):
    return dict(complete=point['positive']['all']['complete'],
                common_events=point['negative']['common']['events'],
                cross_events=point['negative']['cross_target']['events'],
                pre_source=point['pre_source_events'],
                background_events=point['background']['events'])


def compare(ref):
    commit = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    inputs = {}
    def read(name):
        path = BASE + name
        data = subprocess.check_output(['git', 'show', commit + ':' + path])
        inputs[path] = sha(data)
        return json.loads(data)
    comparison = read('4090-phase-aligned-comparison-v1.json')
    grouped = read('4090-phase-aligned-grouped-diagnostic-v1.json')
    audit = read('4090-phase-aligned-scoring-independent-audit-v1.json')
    rejection = read('4090-phase-aligned-diagnostic-rejection-v1.json')
    execution = read('4090-phase-aligned-development-execution-v1.json')
    original = read('4090-hey-common-carrier-raw-development-execution-v1.json')
    assert audit['all_passed'] is True and audit['points'] == 276
    assert grouped['scoring_audit_sha256'] == inputs[BASE + '4090-phase-aligned-scoring-independent-audit-v1.json']
    assert grouped['summary_sha256'] == audit['summary_sha256']
    assert rejection['comparison_sha256'] == inputs[BASE + '4090-phase-aligned-comparison-v1.json']
    for locked in (comparison['files'], grouped['source_hashes']):
        for path, digest in locked.items():
            if path in inputs:
                assert digest == inputs[path], path
    assert execution['grid'] == original['grid']
    assert execution['materials'] == original['materials']
    grid = execution['grid']
    assert len(grid) == len(set(grid)) == 138
    assert all(d[k] is False for d in (comparison, grouped) for k in ('threshold_selected', 'candidate_promoted'))
    assert rejection['selected_threshold'] is None and rejection['candidate_promoted'] is False
    points = {(p['model'], p['threshold']): p for p in comparison['all_new_points']}
    assert len(points) == len(comparison['all_new_points']) == 276
    assert {m['model'] for m in grouped['models']} == {'hey_nyx-nominal', 'hey_nyx-phase'}
    summaries, regions, models = [], [], {}
    worst_far_error = 0.
    for model in grouped['models']:
        name, rows = model['model'], model['points']
        assert [p['threshold'] for p in rows] == grid
        audit_model = next(m for m in audit['models'] if m['model'] == name)
        assert audit_model['points'] == 138 and audit_model['complete_inputs_per_point'] == 372
        assert next(v for p, v in grouped['source_hashes'].items() if p.endswith('/' + name + '.jsonl')) == audit_model['records_sha256']
        models[name] = [points[(name, threshold)] for threshold in grid]
        for row, point in zip(rows, models[name], strict=True):
            assert all(row[k] == point[k] for k in ('positive', 'negative', 'background'))
            bg = point['background']
            assert bg['hours'] == comparison['background_hours'] == 10.483944826388889
            error = abs(poisson_upper95(bg['events'], bg['hours']) - bg['far_upper95'])
            worst_far_error = max(worst_far_error, error)
            assert error < 1e-12
            assert row['far_requirement_satisfied'] == (bg['far_upper95'] <= .5)
            positive = point['positive']
            for parts in (('standalone', 'command'), ('variant0', 'variant1', 'variant2')):
                for key in ('clips', 'any_after_source', 'complete', 'before_only', 'missing'):
                    assert sum(positive[p][key] for p in parts) == positive['all'][key]
            for dimension in DIMENSIONS:
                groups = [g for g in row['groups'] if g['dimension'] == dimension]
                assert sum(g['clips'] for g in groups) == 360
                for role, clips in (('target', 36), ('common', 288), ('reciprocal', 36)):
                    gs = [g for g in groups if g['role'] == role]
                    assert sum(g['clips'] for g in gs) == clips
                    assert all(0 <= g['complete_clips'] <= g['activated_clips'] <= g['clips'] and g['events'] >= g['activated_clips'] for g in gs)
                    if role == 'target':
                        assert sum(g['complete_clips'] for g in gs) == positive['all']['complete']
                    else:
                        neg = point['negative']['common' if role == 'common' else 'cross_target']
                        assert sum(g['activated_clips'] for g in gs) == neg['activated_clips']
                        assert sum(g['events'] for g in gs) == neg['events']
        far = [p for p in models[name] if p['background']['far_upper95'] <= .5]
        joint = [p for p in far if all(metrics(p)[k] == 0 for k in ('common_events', 'cross_events', 'pre_source'))]
        assert model['all_far_qualified_thresholds'] == [p['threshold'] for p in far]
        summaries.append(dict(condition=name.removeprefix('hey_nyx-'), total_points=138,
            far_qualified_points=len(far), far_and_zero_negative_points=len(joint),
            complete_at_all_far_points=span([p['positive']['all']['complete'] for p in far]),
            bare_at_all_far_points=span([p['positive']['standalone']['complete'] for p in far]),
            command_at_all_far_points=span([p['positive']['command']['complete'] for p in far]),
            complete_at_joint_points=span([p['positive']['all']['complete'] for p in joint]),
            minimum_background_far_upper95=min(p['background']['far_upper95'] for p in models[name])))
        for region, selected in (('all138', rows), ('all_far_qualified', [r for r in rows if r['far_requirement_satisfied']])):
            groups = defaultdict(list)
            for row in selected:
                for g in row['groups']:
                    groups[(g['role'], g['dimension'], str(g['value']))].append(g)
            regions.append(dict(model=name, region=region, points=len(selected), groups=[
                dict(role=role, dimension=dimension, value=value, clips=span([g['clips'] for g in gs]),
                     activated=span([g['activated_clips'] for g in gs]),
                     complete=span([g['complete_clips'] for g in gs]), events=span([g['events'] for g in gs]))
                for (role, dimension, value), gs in sorted(groups.items())]))
    assert summaries == comparison['summary']
    paired, original_deltas = [], []
    for nominal, phase in zip(models['hey_nyx-nominal'], models['hey_nyx-phase'], strict=True):
        assert nominal['threshold'] == phase['threshold']
        nm, pm = metrics(nominal), metrics(phase)
        delta = {k: pm[k] - nm[k] for k in nm}
        paired.append(dict(threshold=nominal['threshold'], nominal=nm, phase=pm,
            both_far_qualified=all(p['background']['far_upper95'] <= .5 for p in (nominal, phase)), phase_minus_nominal=delta))
        original_deltas.append(dict(threshold=nominal['threshold'], **{
            'phase_minus_nominal_' + k: v for k, v in delta.items()}))
    assert original_deltas == comparison['all_matched_deltas']
    shared = [p for p in paired if p['both_far_qualified']]
    assert all(s['far_and_zero_negative_points'] == 0 for s in summaries)
    return dict(schema='mac-phase-all-point-comparison-v1', gpu_commit=commit,
        source_sha256=sha(Path(__file__).read_bytes()), input_sha256=inputs,
        checked_points=276, group_dimensions_per_point=6, materials_and_grid_match_original=True,
        maximum_independent_poisson_upper95_error=worst_far_error, recomputed_summary=summaries,
        paired_points=paired, shared_far_points=len(shared),
        all138_delta_signs={k: dict(Counter('higher' if p['phase_minus_nominal'][k] > 0 else 'lower' if p['phase_minus_nominal'][k] < 0 else 'equal' for p in paired)) for k in metrics(models['hey_nyx-nominal'][0])},
        shared_far_delta_ranges={k: span([p['phase_minus_nominal'][k] for p in shared]) for k in metrics(models['hey_nyx-nominal'][0])},
        all_group_regions=regions, threshold_selected=False, candidate_qualified=False,
        model_calls=0, audio_accessed=False, private_trace_oracle_repeated=False,
        limits=['Recomputes committed summaries, all groups, deltas and Poisson bounds; not a private saved-trace/model replay.',
                'Correlated reused source/grid points are not independent speakers or trials. Generated support is not phonetic truth.',
                'No new training, exposure, threshold selection, fresh development, acceptance or P4 qualification.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.gpu_ref)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('checked_points', 'recomputed_summary', 'shared_far_points', 'all138_delta_signs', 'shared_far_delta_ranges', 'maximum_independent_poisson_upper95_error')}))


if __name__ == '__main__':
    main()
