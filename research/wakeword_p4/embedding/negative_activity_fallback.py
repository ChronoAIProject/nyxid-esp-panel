"""Explicit v2 preparation: undefined energy retains original uniform draws."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from negative_activity import interval_map, sha


def uniform_fallback_ids(rows, records, minimum_end):
    """Validate fallback metadata as well as the common interval identities."""
    interval_map(rows, records, minimum_end)
    fallback = set()
    for record in records:
        policy = record.get('sampling_policy', 'source_activity')
        if policy == 'retain_original_uniform_endpoint':
            if (record.get('activity_defined') is not False or
                    record['minimum_end'] != minimum_end or record['maximum_end'] != record['frames']):
                raise ValueError('Undefined-activity fallback requires the entire original warmed range')
            fallback.add(record['id'])
        elif policy != 'source_activity' or record.get('activity_defined', True) is not True:
            raise ValueError('Unknown or contradictory source-activity policy')
    return fallback


def prepare(args):
    if args.output.exists() or args.report.exists():
        raise ValueError('Use fresh v2 manifest/report paths')
    plan = json.loads(args.plan.read_text())
    root = Path(__file__).parent
    for name, expected in plan['preparation_scripts'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen v2 preparation helper differs')
    if sha(args.audit) != plan['training_source_audit_sha256']:
        raise ValueError('Frozen training-source audit differs')
    audit = json.loads(args.audit.read_text())
    if (not audit['all_source_hashes_checked'] or not audit['all_sources_training_negative_only'] or
            not audit['heldout_and_positive_disjoint'] or not audit['source_audio_unchanged'] or
            audit['preparation_plan_sha256'] != plan['failed_v1_preparation_plan_sha256'] or
            audit['v1_failure_sha256'] != plan['v1_failure_sha256']):
        raise ValueError('Invalid original training-only audit')
    sources = {row['source_sha256']: row for row in audit['sources']}
    if len(sources) != len(audit['sources']):
        raise ValueError('Duplicate audited source')
    expected_views = {(source, view['id'], view['_feature_directory'])
                      for source, stats in sources.items() for view in stats['views']}
    rows, records, observed_views = [], [], set()
    fallback_sources, verified_sources = set(), set()
    for directory, binding in plan['feature_inputs'].items():
        base = args.data_root / directory
        for name, expected in binding.items():
            if sha(base / name) != expected:
                raise ValueError('Original feature/source input differs')
        jobs_list = json.loads((base / 'jobs-private.json').read_text())
        jobs = {row['id']: row for row in jobs_list}
        if len(jobs) != len(jobs_list):
            raise ValueError('Duplicate original job')
        for line in (base / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if (row['split'], row['kind'], row['label']) != ('training', 'tts', 0):
                continue
            source = row['source_sha256']
            stats, job = sources[source], jobs[row['id']]
            if any(row[k] != job[k] for k in ('source_sha256', 'engine', 'variant', 'split', 'kind', 'label')):
                raise ValueError('Source identity differs')
            if source not in verified_sources:
                if sha(Path(job['source'])) != source:
                    raise ValueError('Original audited WAV differs')
                verified_sources.add(source)
            view = {k: row[k] for k in ('id', 'features_sha256', 'engine', 'variant', 'frames')} | {'_feature_directory': directory}
            if view not in stats['views'] or abs(row['source_seconds'] - stats['seconds']) > 1e-9:
                raise ValueError('Original audited feature identity or duration differs')
            identity = (source, row['id'], directory)
            if identity in observed_views:
                raise ValueError('Duplicate feature view')
            observed_views.add(identity)
            defined = stats['active_blocks'] > 0
            if defined:
                low = int(np.ceil((2.4 + stats['activity_start_seconds'] + .03) / .08))
                high = int(np.ceil((2.4 + stats['activity_stop_seconds'] + .32) / .08))
            else:
                low, high = 25, row['frames']
                fallback_sources.add(source)
            rows.append(row | {'_feature_directory': directory})
            records.append({k: row[k] for k in ('id', 'source_sha256', 'features_sha256', 'engine', 'variant', 'frames')} |
                {'feature_directory': directory, 'minimum_end': low, 'maximum_end': high,
                 'activity_defined': defined,
                 'sampling_policy': 'source_activity' if defined else 'retain_original_uniform_endpoint'})
    if observed_views != expected_views or verified_sources != set(sources):
        raise ValueError('Audited original sources/views must all remain covered')
    fallback = uniform_fallback_ids(rows, records, 25)
    manifest = {'scope': __doc__, 'preparation_plan_sha256': sha(args.plan),
                'training_source_audit_sha256': sha(args.audit), 'records': records,
                'all_sources_training_only': True, 'heldout_source_hash_disjoint': True,
                'source_audio_unchanged': True, 'audio_speed_changed': False,
                'speaker_playback': False, 'microphone_access': False,
                'source_recordings': len(sources), 'uniform_fallback_sources': len(fallback_sources),
                'uniform_fallback_views': len(fallback), 'accepted': False}
    with args.output.open('x') as out:
        out.write(json.dumps(manifest, indent=2) + '\n')
    groups = defaultdict(list)
    for record in records:
        groups[record['engine']].append(record)
    cohorts = []
    for engine, group in sorted(groups.items()):
        defined = [r for r in group if r['activity_defined']]
        fraction = float(np.mean([(r['maximum_end'] - r['minimum_end'] + 1) / (r['frames'] - 25 + 1)
                                  for r in defined])) if defined else None
        cohorts.append({'engine': engine, 'feature_views': len(group), 'defined_activity_views': len(defined),
                        'uniform_fallback_views': len(group) - len(defined),
                        'uniform_endpoint_fraction_in_defined_interval': fraction,
                        'quarter_focused_expected_fraction_given_defined': .25 + .75 * fraction if fraction is not None else None})
    report = {k: v for k, v in manifest.items() if k != 'records'}
    report.update(manifest_sha256=sha(args.output), feature_views=len(records), cohorts=cohorts,
                  limits=['All undefined-activity sources and environmental views retain their ORIGINAL endpoint and consume no new activity-RNG draw.',
                          'Defined interval membership is not speech occupancy or phonetic truth. Undefined cases have no claimed speech span.',
                          'No model/scoring, waveform/features/labels changed, or hardware/audio-device access.'])
    with args.report.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'plan', 'audit', 'output', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(parser.parse_args())
