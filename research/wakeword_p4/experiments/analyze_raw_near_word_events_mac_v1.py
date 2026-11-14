"""Describe every retained paired-raw event against original generated supports.

This consumes authenticated diagnostic JSONLs, never models or audio. Generated
support intervals are not phonetic annotations and cannot prove early waking.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess

BASE = 'research/wakeword_p4/results/'
BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample',
              'generated_container_end_sample', 'full_rir_container_end_sample')
BANDS = ('before_source', 'within_generated_prefix',
         'prefix_to_generated_container', 'at_or_after_generated_container')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, 'Duplicate JSON field: ' + key
        result[key] = value
    return result


def load(data):
    return json.loads(data, object_pairs_hook=unique_object)


def band(sample, row):
    if sample < row['lead_samples']:
        return BANDS[0]
    if sample < row['generated_prefix_end_sample']:
        return BANDS[1]
    if sample < row['generated_container_end_sample']:
        return BANDS[2]
    return BANDS[3]


def check_events(observed, row, synthetic):
    events = observed['events']
    assert isinstance(events, list)
    previous = 0
    last_step = (max(0, (row['samples'] + 159) // 160 - 2) + 2) // 3
    for event in events:
        step, sample = event['step'], event['sample']
        assert type(step) is int and previous < step <= last_step and step >= 5
        assert type(sample) is int and sample == min((3 * step + 2) * 160, row['samples'])
        previous = step
        if synthetic:
            for boundary in BOUNDARIES:
                assert event[boundary + '_signed_ms'] == (sample - row[boundary]) / 16
    return [event['sample'] for event in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    for name in ('events-dir', 'synthetic', 'background', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    inputs = {}

    def read(name):
        path = BASE + name
        data = subprocess.check_output(['git', 'show', commit + ':' + path])
        inputs[path] = sha(data)
        return load(data)

    execution = read('4090-hey-common-carrier-raw-development-execution-v1.json')
    audit = read('4090-hey-common-carrier-raw-scoring-independent-audit-v1.json')
    summary = read('4090-hey-common-carrier-raw-diagnostic-summary-v1.json')
    grouped = read('4090-hey-common-carrier-raw-grouped-diagnostic-v1.json')
    assert audit['all_passed'] and audit['points'] == 552
    assert summary['source_summary_sha256'] == audit['summary_sha256'] == grouped['summary_sha256']
    assert summary['execution_sha256'] == inputs[BASE + '4090-hey-common-carrier-raw-development-execution-v1.json']
    assert grouped['scoring_audit_sha256'] == inputs[BASE + '4090-hey-common-carrier-raw-scoring-independent-audit-v1.json']
    materials = {}
    for phase in ('synthetic', 'background'):
        raw = getattr(args, phase).read_bytes()
        assert sha(raw) == execution['materials'][phase]['sha256']
        materials[phase] = load(raw)['records']
        assert len(materials[phase]) == execution['materials'][phase]['records']
        inputs[phase + '_manifest'] = sha(raw)
    assert len(materials['synthetic']) == 360 and len(materials['background']) == 12
    expected = {(point['model'], point['threshold']): point for point in summary['points']}
    assert len(expected) == 552 and len(execution['models']) == 4
    grid = execution['grid']
    assert len(grid) == len(set(grid)) == 138
    results = []
    for model in execution['models']:
        name = model['target'] + '-' + model['condition']
        data = (args.events_dir / (name + '.jsonl')).read_bytes()
        digest = sha(data)
        assert digest == next(item['records_sha256'] for item in audit['models'] if item['model'] == name)
        assert digest == next(value for key, value in grouped['source_hashes'].items() if key.endswith('/' + name + '.jsonl'))
        inputs[name + '.jsonl'] = digest
        points = [load(line) for line in data.splitlines()]
        assert len(points) == 138 and [point['summary']['threshold'] for point in points] == grid
        for point in points:
            s = point['summary']
            assert s == expected[(name, s['threshold'])] and len(point['records']) == 372
            groups = {}
            negative = {key: dict(clips=0, activated_clips=0, events=0) for key in ('common', 'cross_target')}
            positive_complete = positive_after = pre_source = bg_events = 0
            position = 0
            for phase, rows in materials.items():
                for index, row in enumerate(rows):
                    observed = point['records'][position]
                    position += 1
                    assert (observed['phase'], observed['index'], observed['source_id']) == (phase, index, row['source_id'])
                    times = check_events(observed, row, phase == 'synthetic')
                    if phase == 'background':
                        bg_events += len(times)
                        continue
                    assert all(observed[key] == row[key] for key in ('target_id', 'variant', 'carrier', 'length_scale'))
                    role = 'target' if row['target_id'] == model['target'] else (
                        'cross_target' if row['target_id'] in ('hey_willow', 'hey_nyx') else 'common')
                    if role == 'target':
                        after = [sample for sample in times if sample >= row['lead_samples']]
                        complete = any(sample >= row['generated_prefix_end_sample'] for sample in after)
                        assert observed['positive']['any_after_source'] == bool(after)
                        assert observed['positive']['complete'] == complete
                        positive_complete += complete
                        positive_after += bool(after)
                        pre_source += sum(sample < row['lead_samples'] for sample in times)
                    else:
                        negative[role]['clips'] += 1
                        negative[role]['activated_clips'] += bool(times)
                        negative[role]['events'] += len(times)
                    key = (role, row['literal'], row['carrier'])
                    group = groups.setdefault(key, dict(role=role, literal=row['literal'], carrier=row['carrier'],
                        clips=0, activated_clips=0, events=0,
                        events_by_support_band={key: 0 for key in BANDS},
                        clips_by_support_band={key: 0 for key in BANDS}, first_signed_ms=[]))
                    group['clips'] += 1
                    group['activated_clips'] += bool(times)
                    group['events'] += len(times)
                    bands = [band(sample, row) for sample in times]
                    for key in BANDS:
                        group['events_by_support_band'][key] += bands.count(key)
                        group['clips_by_support_band'][key] += key in bands
                    if times:
                        group['first_signed_ms'].append((times[0] - row['generated_prefix_end_sample']) / 16)
            assert negative == s['negative'] and positive_complete == s['positive']['all']['complete']
            assert positive_after == s['positive']['all']['any_after_source'] and pre_source == s['pre_source_events']
            hours = sum(row['samples'] / 16000 / 3600 for row in materials['background'])
            assert bg_events == s['background']['events'] and math.isclose(hours, s['background']['hours'], rel_tol=1e-14)
            for group in groups.values():
                times = group.pop('first_signed_ms')
                group['first_event_minus_generated_prefix_ms'] = dict(observed=len(times),
                    minimum=min(times) if times else None, median=median(times) if times else None,
                    maximum=max(times) if times else None)
            results.append(dict(model=name, threshold=s['threshold'],
                far_requirement_satisfied=s['background']['far_upper95'] <= .5,
                background=s['background'], groups=[groups[key] for key in sorted(groups)]))
    assert len(results) == 552
    result = dict(schema='mac-original-paired-raw-support-timing-v1', gpu_commit=commit,
        source_sha256=sha(Path(__file__).read_bytes()), input_sha256=inputs,
        checked_points=552, records_per_point=372, all_original_jsonl_hashes_matched=True,
        all_synthetic_and_background_event_coordinates_checked=True, points=results,
        model_calls=0, audio_accessed=False, threshold_selected=False, candidate_qualified=False,
        limits=[
            'Authenticated original event JSONLs and manifests only; no local raw-probability oracle or model replay.',
            'All original points and every source retained. Support bands are generated allocation intervals, not phonetic truth or measured live latency.',
            'Repeated thresholds, variants and single Semaine voice are correlated. These descriptive timings do not establish cause, FAR qualification, INT8 or independent acceptance.'])
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(points=552, original_files=4, event_coordinates_checked=True, new_model_calls=0)))


if __name__ == '__main__':
    main()
