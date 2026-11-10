"""Prepare bounded training endpoint overrides; preserve original audio/features.

The rule applies to complete standalone aliases in original and environmental
views. It does not require a marker to equal file length. Validation, commands,
negative labels and original metadata stay intact; no model is fitted here.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import wave

import numpy as np

from audit_ctc_endpoints import sha, standalone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root.resolve()
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if sha(Path(__file__)) != plan['script_sha256'] or sha(Path(__file__).with_name('audit_ctc_endpoints.py')) != plan['helper_sha256']:
        raise ValueError('Preparation helper differs from frozen plan')
    for name, expected in plan['inputs'].items():
        if sha(base / name) != expected:
            raise ValueError('Frozen input differs: ' + name)
    metadata_path = (base / plan['text_metadata']).resolve()
    metadata = defaultdict(list)
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] == 0:
            metadata[(metadata_path.parent / 'dataset' / row['path']).resolve()].append(row)
    directories, heldout, original_ids = {}, set(), set()
    for directory in plan['feature_directories']:
        path = base / directory
        rows = [json.loads(line) for line in (path / 'index.jsonl').read_text().splitlines()]
        jobs_list = json.loads((path / 'jobs-private.json').read_text())
        jobs = {job['id']: job for job in jobs_list}
        if len(jobs) != len(jobs_list) or len(rows) != len(jobs) or {r['id'] for r in rows} != set(jobs):
            raise ValueError('Feature/job coverage differs')
        for row in rows:
            if row['id'] in original_ids:
                raise ValueError('Repeated feature identity across source directories')
            original_ids.add(row['id'])
            if row['split'] != 'training':
                heldout.add(row['source_sha256'])
        directories[directory] = (rows, jobs)
    cache, overrides = {}, []
    total_rows, preserved_nontraining = 0, 0
    for directory, (rows, jobs) in directories.items():
        total_rows += len(rows)
        for row in rows:
            if row['split'] != 'training':
                preserved_nontraining += 1
                continue
            if row['kind'] != 'tts' or row['label'] != 1:
                continue
            job = jobs[row['id']]
            if (job['split'], job['kind'], job['label'], job['source_sha256'], job['variant'], job['engine']) != (
                    'training', 'tts', 1, row['source_sha256'], row['variant'], row['engine']):
                raise ValueError('Training feature/job role differs')
            if row['source_sha256'] in heldout:
                raise ValueError('Training source overlaps held-out audio')
            expected_end = (plan['lead_samples'] + job['keyword_end']) / 16000
            if row['keyword_end'] != expected_end:
                raise ValueError('Original feature marker differs from source job')
            source = Path(job['source']).resolve()
            entries = metadata[source]
            if not entries or any(
                (r['split'], r['label'], r['sha256'], r['engine'], r['keyword_end']) !=
                ('training', 1, row['source_sha256'], row['engine'], job['keyword_end'])
                for r in entries
            ):
                raise ValueError('Positive metadata role/hash/marker differs')
            texts = sorted({r['text'] for r in entries})
            if not all(standalone(text) for text in texts):
                continue
            key = row['source_sha256']
            if key not in cache:
                if sha(source) != key:
                    raise ValueError('Source PCM hash differs')
                with wave.open(str(source), 'rb') as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                        raise ValueError('Expected original 16 kHz mono PCM16 source')
                    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
                padded = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
                rms = np.sqrt(np.mean(padded.reshape(-1, 160) ** 2, axis=1))
                active = np.flatnonzero(rms >= max(float(rms.max()) * plan['rms_peak_fraction'], 1.))
                if not len(active):
                    raise ValueError('Selected positive source has no activity')
                end = min(len(pcm), int((active[-1] + 1) * 160))
                late = job['keyword_end'] - end
                eligible = late > plan['minimum_late_samples']
                corrected = min(job['keyword_end'], end + plan['post_activity_margin_samples']) if eligible else job['keyword_end']
                cache[key] = {'source_sha256': key, 'engine': row['engine'], 'texts': texts,
                              'source_samples': len(pcm), 'original_keyword_end_samples': job['keyword_end'],
                              'activity_end_samples': end, 'eligible': eligible,
                              'previous_file_end_eligible': eligible and job['keyword_end'] == len(pcm),
                              'corrected_samples': corrected}
            source_record = cache[key]
            if (source_record['engine'], source_record['original_keyword_end_samples']) != (row['engine'], job['keyword_end']):
                raise ValueError('Conflicting same-audio source marker')
            if source_record['source_samples'] != round(row['source_seconds'] * 16000):
                raise ValueError('Source length differs from feature metadata')
            if source_record['eligible']:
                feature = Path(row['path'])
                feature = feature if feature.is_absolute() else base / feature
                if sha(feature) != row['features_sha256']:
                    raise ValueError('Affected feature payload differs')
                overrides.append({
                    'id': row['id'], 'feature_directory': directory, 'engine': row['engine'],
                    'source_sha256': key, 'features_sha256': row['features_sha256'],
                    'variant': row['variant'], 'old_keyword_end': row['keyword_end'],
                    'new_keyword_end': (plan['lead_samples'] + source_record['corrected_samples']) / 16000,
                })
    count = len(cache)
    eligible = sum(r['eligible'] for r in cache.values())
    old_eligible = sum(r['previous_file_end_eligible'] for r in cache.values())
    if (count, eligible, old_eligible, len(overrides)) != tuple(plan['expected_counts']):
        raise ValueError('Preparation differs from the recorded all-training source accounting')
    args.output.mkdir(parents=True)
    payload = {'scope': __doc__, 'preparation_plan_sha256': sha(args.plan),
               'source_feature_indexes': {name: plan['inputs'][name + '/index.jsonl'] for name in plan['feature_directories']},
               'records': overrides}
    payload_path = args.output / 'training-endpoints.json'
    payload_path.write_text(json.dumps(payload, indent=2) + '\n')
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'script_sha256': sha(Path(__file__)),
              'source_rows': total_rows, 'preserved_nontraining_rows': preserved_nontraining,
              'standalone_sources': count, 'eligible_sources': eligible,
              'previous_file_end_eligible_sources': old_eligible,
              'changed_training_endpoints': len(overrides),
              'changed_rows_by_directory': dict(Counter(r['feature_directory'] for r in overrides)),
              'changed_sources_by_engine': dict(Counter(r['engine'] for r in cache.values() if r['eligible'])),
              'overrides_sha256': sha(payload_path),
              'original_indexes_and_payloads_changed': False, 'heldout_audio_loaded': False,
              'audio_speed_changed': False, 'accepted': False}
    (args.output / 'eligibility-rows.json').write_text(json.dumps(list(cache.values()), indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
