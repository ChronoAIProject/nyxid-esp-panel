"""Independently compare every published paired raw point, without selecting one.

Reads committed summaries/groups and their locks, not audio, model or private
saved traces. Does not duplicate the GPU's independent raw event oracle.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'research/wakeword_p4/results/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def span(values):
    return [min(values), max(values)] if values else None


def compare(ref):
    commit = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    inputs = {}
    def read(name):
        path = BASE + name
        data = subprocess.check_output(['git', 'show', commit + ':' + path])
        inputs[path] = sha(data)
        return json.loads(data)
    summary = read('4090-hey-common-carrier-raw-diagnostic-summary-v1.json')
    grouped = read('4090-hey-common-carrier-raw-grouped-diagnostic-v1.json')
    audit = read('4090-hey-common-carrier-raw-scoring-independent-audit-v1.json')
    execution = read('4090-hey-common-carrier-raw-development-execution-v1.json')
    original = read('4090-fp32-reused-development-execution-v1.json')
    assert summary['execution_sha256'] == inputs[BASE + '4090-hey-common-carrier-raw-development-execution-v1.json']
    assert audit['all_passed'] and audit['points'] == 552
    assert grouped['scoring_audit_sha256'] == inputs[BASE + '4090-hey-common-carrier-raw-scoring-independent-audit-v1.json']
    assert grouped['summary_sha256'] == audit['summary_sha256'] == summary['source_summary_sha256']
    assert execution['materials'] == original['materials'] and execution['grid'] == original['grid']
    for path, digest in grouped['source_hashes'].items():
        if path in inputs:
            assert inputs[path] == digest, path
    grid = execution['grid']
    assert len(grid) == len(set(grid)) == 138
    points = {(row['model'], row['threshold']): row for row in summary['points']}
    assert len(points) == 552
    models, inventories = {}, []
    for model in grouped['models']:
        name = model['model']
        rows = model['points']
        assert [p['threshold'] for p in rows] == grid
        assert model['all_far_qualified_thresholds'] == [p['threshold'] for p in rows if p['far_requirement_satisfied']]
        expected_hash = next(row['records_sha256'] for row in audit['models'] if row['model'] == name)
        assert next(value for path, value in grouped['source_hashes'].items() if path.endswith('/' + name + '.jsonl')) == expected_hash
        for point in rows:
            raw = points[(name, point['threshold'])]
            assert all(point[key] == raw[key] for key in ('background', 'positive', 'negative'))
            assert point['far_requirement_satisfied'] == (point['background']['far_upper95'] <= .5)
            for dimension in ('carrier_kind', 'literal', 'voice', 'source_id', 'length_scale', 'variant'):
                groups = [g for g in point['groups'] if g['dimension'] == dimension]
                assert sum(g['clips'] for g in groups) == 360
                assert sum(g['clips'] for g in groups if g['role'] == 'target') == 36
                assert sum(g['clips'] for g in groups if g['role'] == 'common') == 288
                assert sum(g['clips'] for g in groups if g['role'] == 'reciprocal') == 36
                assert sum(g['complete_clips'] for g in groups) == point['positive']['all']['complete']
                for role, key in (('common', 'common'), ('reciprocal', 'cross_target')):
                    assert sum(g['activated_clips'] for g in groups if g['role'] == role) == point['negative'][key]['activated_clips']
                    assert sum(g['events'] for g in groups if g['role'] == role) == point['negative'][key]['events']
        models[name] = rows
        regions = []
        for region, selected in (('all138', rows), ('far_upper95_le_half', [p for p in rows if p['far_requirement_satisfied']])):
            literal, carrier = defaultdict(list), defaultdict(list)
            for point in selected:
                for group in point['groups']:
                    if group['role'] != 'common':
                        continue
                    if group['dimension'] == 'literal':
                        literal[group['value']].append(group['activated_clips'])
                    if group['dimension'] == 'carrier_kind':
                        carrier[group['value']].append(group['activated_clips'])
            regions.append(dict(region=region, points=len(selected),
                complete=span([p['positive']['all']['complete'] for p in selected]),
                common_activated=span([p['negative']['common']['activated_clips'] for p in selected]),
                cross_activated=span([p['negative']['cross_target']['activated_clips'] for p in selected]),
                common_by_literal={key: span(value) for key, value in sorted(literal.items())},
                common_by_carrier={key: span(value) for key, value in sorted(carrier.items())}))
        inventories.append(dict(model=name, regions=regions))
    pairs = []
    for target in ('hey_willow', 'hey_nyx'):
        control, treatment = models[target + '-control'], models[target + '-command_context']
        paired = []
        for c, t in zip(control, treatment, strict=True):
            assert c['threshold'] == t['threshold']
            assert c['background']['hours'] == t['background']['hours'] == 10.483944826388889
            def metrics(p):
                raw = points[(target + ('-control' if p is c else '-command_context'), p['threshold'])]
                return dict(complete=p['positive']['all']['complete'],
                    any_after_source=p['positive']['all']['any_after_source'],
                    common_activated=p['negative']['common']['activated_clips'],
                    cross_activated=p['negative']['cross_target']['activated_clips'],
                    pre_source=raw['pre_source_events'], background_events=p['background']['events'])
            cm, tm = metrics(c), metrics(t)
            delta = {key: tm[key] - cm[key] for key in cm}
            comparable = c['far_requirement_satisfied'] and t['far_requirement_satisfied']
            def dominates(a, b):
                comparisons = [a['complete'] >= b['complete']] + [a[k] <= b[k] for k in ('common_activated', 'cross_activated', 'pre_source', 'background_events')]
                strict = a['complete'] > b['complete'] or any(a[k] < b[k] for k in ('common_activated', 'cross_activated', 'pre_source', 'background_events'))
                return all(comparisons) and strict
            paired.append(dict(threshold=c['threshold'], both_far_qualified=comparable,
                control=cm, treatment=tm, treatment_minus_control=delta,
                treatment_dominates=dominates(tm, cm), control_dominates=dominates(cm, tm)))
        selected = [p for p in paired if p['both_far_qualified']]
        pairs.append(dict(target=target, points=paired, shared_far_points=len(selected),
            shared_far_delta_ranges={key: span([p['treatment_minus_control'][key] for p in selected]) for key in paired[0]['control']},
            shared_far_tradeoff_counts=dict(treatment_dominates=sum(p['treatment_dominates'] for p in selected),
                control_dominates=sum(p['control_dominates'] for p in selected),
                neither_dominates=sum(not p['treatment_dominates'] and not p['control_dominates'] for p in selected)),
            all138_complete_delta_signs=dict(Counter('higher' if p['treatment_minus_control']['complete'] > 0 else 'lower' if p['treatment_minus_control']['complete'] < 0 else 'same' for p in paired))))
    return dict(schema='mac-raw-common-carrier-all-point-comparison-v1', gpu_commit=commit,
        source_sha256=sha(Path(__file__).read_bytes()), input_sha256=inputs,
        checked_summary_points=552, checked_group_dimensions_per_point=6, original_materials_and_grid_match=True,
        descriptive_regions=inventories, paired_comparisons=pairs,
        threshold_selected=False, models_invoked=0, audio_accessed=False,
        independent_private_trace_oracle_repeated=False, candidate_qualified=False,
        limits=[
            'Independent committed-summary/group consistency and matched-point arithmetic only; no local private model/trace or raw event reduction.',
            'All thresholds retained; correlated grid points and one Semaine voice are not independent trials or speaker coverage.',
            'Dominance refers only to generated-boundary complete count and listed errors at the identical numeric threshold, not a chosen workpoint or maturity.',
            'The training intervention jointly changes command context, duration/crop and stochastic realization; it is not a text-only causal contrast.',
            'Zero-negative/FAR conjunction remains unsatisfied; no fresh development, independent acceptance, INT8 or full-panel acceptance.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.gpu_ref)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(points=report['checked_summary_points'], pairs=[{k:v for k,v in pair.items() if k!='points'} for pair in report['paired_comparisons']])))


if __name__ == '__main__':
    main()
