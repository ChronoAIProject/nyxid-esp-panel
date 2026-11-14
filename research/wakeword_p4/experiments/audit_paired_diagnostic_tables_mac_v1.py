"""Audit all committed four-arm tables without selecting thresholds or invoking models.

The historical raw-context and frozen-encoder profiles cover committed tables.
Neither profile repeats a private trace oracle.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

BASE = 'research/wakeword_p4/results/'
DIMENSIONS = ('carrier_kind', 'literal', 'voice', 'source_id', 'length_scale', 'variant')
PROFILES = {
    'raw-context': ('4090-hey-common-carrier-raw-', 'diagnostic-summary', 'points', ('control', 'command_context')),
    'frozen-encoder': ('4090-frozen-encoder-', 'comparison', 'all_new_points', ('full', 'frozen')),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def span(values):
    return [min(values), max(values)] if values else [None, None]


def poisson_upper95(count, hours):
    """Invert the Poisson CDF directly, independently of SciPy/chi-square.

    The upper endpoint is above count. Sum downwards from P(X=count) in
    log space, so larger event counts do not underflow at P(X=0).
    """
    assert type(count) is int and count >= 0 and math.isfinite(hours) and hours > 0

    def cdf(mu):
        if count == 0:
            return math.exp(-mu)
        term, terms = 1., [1.]
        for k in range(count, 0, -1):
            term *= k / mu
            terms.append(term)
            if term < 1e-17:
                break
        return math.exp(-mu + count * math.log(mu) - math.lgamma(count + 1)
                        + math.log(math.fsum(terms)))

    low, high = float(count), count + max(8., 10. * math.sqrt(count + 1))
    while cdf(high) > .05:
        high *= 2
    for _ in range(80):
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


def audit_tables(points, grouped, audit, execution, conditions):
    grid = execution['grid']
    assert len(grid) == len(set(grid)) == 138
    assert all(math.isfinite(t) and 0 < t <= 1 for t in grid)
    names = {t + '-' + c for t in ('hey_nyx', 'hey_willow') for c in conditions}
    assert len(names) == 4 and len(points) == 552
    assert len(grouped['models']) == len(audit['models']) == 4
    assert {m['model'] for m in grouped['models']} == {m['model'] for m in audit['models']} == names
    assert audit['all_passed'] is True and audit['points'] == 552
    indexed = {(p['model'], p['threshold']): p for p in points}
    assert len(indexed) == 552 and {p['model'] for p in points} == names
    assert grouped['threshold_selected'] is False and grouped['candidate_promoted'] is False
    models, summaries, regions = {}, [], []
    worst_error, hours = 0., None
    for model in grouped['models']:
        name, rows = model['model'], model['points']
        assert [r['threshold'] for r in rows] == grid
        proof = next(m for m in audit['models'] if m['model'] == name)
        assert proof['points'] == 138 and proof['complete_inputs_per_point'] == 372
        hashes = [v for p, v in grouped['source_hashes'].items() if p.endswith('/' + name + '.jsonl')]
        assert hashes == [proof['records_sha256']]
        models[name] = [indexed[(name, t)] for t in grid]
        for row, point in zip(rows, models[name], strict=True):
            assert point['threshold_selected'] is False
            assert all(row[k] == point[k] for k in ('positive', 'negative', 'background'))
            bg = point['background']
            hours = bg['hours'] if hours is None else hours
            assert bg['hours'] == hours
            upper = poisson_upper95(bg['events'], hours)
            error = abs(upper - bg['far_upper95'])
            assert error <= 2e-10 * max(1., upper)
            worst_error = max(worst_error, error)
            assert row['far_requirement_satisfied'] == (bg['far_upper95'] <= .5) == (upper <= .5)
            positive = point['positive']
            assert positive['all']['clips'] == 36
            for parts in (('standalone', 'command'), ('variant0', 'variant1', 'variant2')):
                for key in ('clips', 'any_after_source', 'complete', 'before_only', 'missing'):
                    assert sum(positive[p][key] for p in parts) == positive['all'][key]
            for dimension in DIMENSIONS:
                groups = [g for g in row['groups'] if g['dimension'] == dimension]
                assert {g['role'] for g in groups} == {'target', 'common', 'reciprocal'}
                for role, clips in (('target', 36), ('common', 288), ('reciprocal', 36)):
                    gs = [g for g in groups if g['role'] == role]
                    assert sum(g['clips'] for g in gs) == clips
                    assert all(0 <= g['complete_clips'] <= g['activated_clips'] <= g['clips']
                               and g['events'] >= g['activated_clips'] for g in gs)
                    if role == 'target':
                        assert sum(g['complete_clips'] for g in gs) == positive['all']['complete']
                    else:
                        neg = point['negative']['common' if role == 'common' else 'cross_target']
                        assert neg['clips'] == clips
                        assert sum(g['activated_clips'] for g in gs) == neg['activated_clips']
                        assert sum(g['events'] for g in gs) == neg['events']
            assert {g['dimension'] for g in row['groups']} == set(DIMENSIONS)
        far = [p for p in models[name] if p['background']['far_upper95'] <= .5]
        joint = [p for p in far if all(metrics(p)[k] == 0 for k in ('common_events', 'cross_events', 'pre_source'))]
        assert model['all_far_qualified_thresholds'] == [p['threshold'] for p in far]
        summaries.append(dict(model=name, far_qualified_points=len(far),
            joint_qualified_points=len(joint),
            complete_at_all_far_points=span([p['positive']['all']['complete'] for p in far]),
            bare_at_all_far_points=span([p['positive']['standalone']['complete'] for p in far]),
            command_at_all_far_points=span([p['positive']['command']['complete'] for p in far]),
            minimum_far_upper95=min(p['background']['far_upper95'] for p in models[name]),
            complete_at_joint_points=span([p['positive']['all']['complete'] for p in joint])))
        for region, selected in (('all138', rows), ('all_far_qualified', [r for r in rows if r['far_requirement_satisfied']])):
            buckets = defaultdict(list)
            for row in selected:
                for g in row['groups']:
                    buckets[(g['role'], g['dimension'], json.dumps(g['value'], sort_keys=True))].append(g)
            regions.append(dict(model=name, region=region, points=len(selected), groups=[
                dict(role=role, dimension=dim, value=json.loads(value),
                     **{k: span([g[k] for g in gs]) for k in ('clips', 'activated_clips', 'complete_clips', 'events')})
                for (role, dim, value), gs in sorted(buckets.items())]))
    pairs = []
    for target in ('hey_nyx', 'hey_willow'):
        for a, b in zip(models[target + '-' + conditions[0]], models[target + '-' + conditions[1]], strict=True):
            am, bm = metrics(a), metrics(b)
            pairs.append(dict(target=target, threshold=a['threshold'],
                both_far_qualified=all(p['background']['far_upper95'] <= .5 for p in (a, b)),
                first=am, second=bm, second_minus_first={k: bm[k] - am[k] for k in am}))
    return dict(checked_points=552, group_dimensions_per_point=6,
                maximum_independent_poisson_upper95_error=worst_error,
                recomputed_summary=summaries, all_matched_points=pairs,
                all_group_regions=regions, background_hours=hours)


class NotReady(Exception):
    pass


def audit_loaded(profile, table, grouped, audit, execution, original, inputs):
    prefix, _, field, conditions = PROFILES[profile]
    assert execution['materials'] == original['materials'] and execution['grid'] == original['grid']
    assert grouped['scoring_audit_sha256'] == inputs[BASE + prefix + 'scoring-independent-audit-v1.json']
    assert grouped['summary_sha256'] == audit['summary_sha256']
    if profile == 'raw-context':
        assert table['source_summary_sha256'] == audit['summary_sha256']
        assert table['execution_sha256'] == inputs[BASE + prefix + 'development-execution-v1.json']
        assert all(table[k] is False for k in ('threshold_selected', 'acceptance_scored', 'candidate_promoted'))
    else:
        assert table['selected_threshold'] is None and table['candidate_promoted'] is False
        hashes = [v for p, v in table['files'].items() if p.endswith('/diagnostic/summary.json')]
        assert hashes == [audit['summary_sha256']]
    for manifest in (table.get('files', {}), grouped['source_hashes']):
        for path, digest in manifest.items():
            if path in inputs:
                assert inputs[path] == digest, path
    result = audit_tables(table[field], grouped, audit, execution, conditions)
    if profile == 'frozen-encoder':
        declared = {(r['target'] + '-' + r['condition']): r for r in table['summary']}
        assert len(table['summary']) == len(declared) == 4
        for row in result['recomputed_summary']:
            # The actual frozen reporter names this region "all_joint"; the
            # earlier labelled old-data fixture used the checker's shorter key.
            # Match the real schema explicitly, including all additional ranges.
            expected = {k: v for k, v in row.items() if k != 'model'}
            expected['complete_at_all_joint_points'] = expected.pop('complete_at_joint_points')
            target, condition = row['model'].rsplit('-', 1)
            assert dict(target=target, condition=condition, **expected) == declared[row['model']]
        deltas = []
        for pair in result['all_matched_points']:
            d = pair['second_minus_first']
            deltas.append(dict(target=pair['target'], threshold=pair['threshold'],
                frozen_minus_full_complete=d['complete'],
                frozen_minus_full_background_events=d['background_events'],
                frozen_minus_full_pre_source=d['pre_source'],
                frozen_minus_full_common_events=d['common_events'],
                frozen_minus_full_cross_target_events=d['cross_events']))
        assert deltas == table['all_matched_deltas']
        rejected = sorted(r['model'] for r in result['recomputed_summary'] if r['joint_qualified_points'] == 0)
        assert rejected == sorted(table['rejected_for_no_joint_point'])
    return result


def compare(ref, profile):
    commit = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    inputs = {}

    def read(name):
        path = BASE + name
        r = subprocess.run(['git', 'show', commit + ':' + path], capture_output=True)
        if r.returncode:
            raise NotReady(path)
        inputs[path] = sha(r.stdout)
        return json.loads(r.stdout)

    prefix, table_name, _, conditions = PROFILES[profile]
    table = read(prefix + table_name + '-v1.json')
    grouped = read(prefix + 'grouped-diagnostic-v1.json')
    audit = read(prefix + 'scoring-independent-audit-v1.json')
    execution = read(prefix + 'development-execution-v1.json')
    original = read('4090-fp32-reused-development-execution-v1.json')
    result = audit_loaded(profile, table, grouped, audit, execution, original, inputs)
    return dict(schema='mac-paired-diagnostic-tables-audit-v1', profile=profile,
        gpu_commit=commit, source_sha256=sha(Path(__file__).read_bytes()), input_sha256=inputs,
        conditions=list(conditions), **result, threshold_selected=False, candidate_qualified=False,
        model_calls=0, audio_accessed=False, private_trace_oracle_repeated=False,
        limits=['Committed-table/hash/group/arithmetic audit only; no private trace or saved-model replay.',
                'All138 points per model retained; correlated reused sources are not independent speakers or acceptance.',
                'Joint diagnostic points, including silent detectors, do not qualify a candidate.',
                'No new data, training, selection, fresh development/acceptance or P4/audio lifecycle qualification.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.gpu_ref, args.profile)
    except NotReady as e:
        print(json.dumps(dict(status='not_ready', missing=str(e), models_invoked=0)))
        return 2
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('profile', 'checked_points', 'recomputed_summary', 'maximum_independent_poisson_upper95_error')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
