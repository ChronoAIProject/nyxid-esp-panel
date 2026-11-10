"""Create a training-only metadata view without changing any feature/audio bytes.

Only the existing extra environmental views are eligible. Base features and
representative sampling stay unchanged, isolating the supervised endpoint change.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if sha(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Preparation script differs')
    for file, expected in plan['inputs'].items():
        if sha(base / file) != expected:
            raise ValueError('Frozen input differs: ' + file)
    metadata_path = (base / plan['text_metadata']).resolve()
    metadata = defaultdict(list)
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] == 0:
            metadata[(metadata_path.parent / 'dataset' / row['path']).resolve()].append(row)
    reference = base / plan['reference_directory']
    original = [json.loads(line) for line in (reference / 'index.jsonl').read_text().splitlines()]
    jobs = {j['id']: j for j in json.loads((reference / 'jobs-private.json').read_text())}
    if len(original) != 20140 or any(r['split'] != 'training' or r['kind'] != 'tts' for r in original):
        raise ValueError('Expected complete original training environment views')
    base_rows = [json.loads(line) for line in (base / 'head-features-tts-v1/index.jsonl').read_text().splitlines()]
    heldout = {r['source_sha256'] for r in base_rows if r['split'] != 'training'}
    if heldout.intersection(r['source_sha256'] for r in original):
        raise ValueError('Held-out source in training views')
    cache, replacements, changed = {}, {}, []
    rows, output_jobs = [], []
    for row in original:
        job = jobs[row['id']]
        updated, updated_job = dict(row), dict(job)
        expected_marker = (plan['lead_samples'] + job['keyword_end']) / 16000 if row['label'] else None
        if row['keyword_end'] != expected_marker:
            raise ValueError('Original endpoint metadata differs')
        if row['label']:
            source = Path(job['source']).resolve()
            entries = metadata[source]
            if not entries or any(e['split'] != 'training' or e['label'] != 1
                                  or e['sha256'] != row['source_sha256'] for e in entries):
                raise ValueError('Positive source metadata role differs')
            texts = sorted({e['text'] for e in entries})
            standalone = all(re.findall('[a-z]+', text.lower()) in
                             [['hey', 'nyx'], ['hey', 'nix'], ['hey', 'nicks']] for text in texts)
            source_samples = round(row['source_seconds'] * 16000)
            if standalone and job['keyword_end'] == source_samples:
                key = row['source_sha256']
                if key not in cache:
                    if sha(source) != key:
                        raise ValueError('Positive source waveform differs')
                    pcm, rate = sf.read(source, dtype='int16')
                    if rate != 16000 or pcm.ndim != 1 or len(pcm) != source_samples:
                        raise ValueError('Unexpected source format')
                    padded = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
                    rms = np.sqrt(np.mean(padded.reshape(-1, 160) ** 2, axis=1))
                    active = np.flatnonzero(rms >= max(float(rms.max()) * plan['rms_peak_fraction'], 1.))
                    if not len(active):
                        raise ValueError('Positive source has no activity')
                    cache[key] = min(len(pcm), int(active[-1] + 1) * 160)
                endpoint = cache[key]
                if source_samples - endpoint > plan['minimum_late_samples']:
                    corrected = min(source_samples, endpoint + plan['post_activity_margin_samples'])
                    if corrected >= source_samples or corrected <= 0:
                        raise ValueError('Invalid corrected positive endpoint')
                    updated['keyword_end'] = (plan['lead_samples'] + corrected) / 16000
                    updated_job['keyword_end'] = corrected
                    updated_job['original_keyword_end'] = job['keyword_end']
                    replacements[key] = {'engine': row['engine'], 'source_sha256': key,
                        'original_samples': source_samples, 'activity_end_samples': endpoint,
                        'corrected_samples': corrected, 'texts': texts}
                    changed.append({'id': row['id'], 'engine': row['engine'],
                        'source_sha256': key, 'variant': row['variant'],
                        'old_keyword_end': row['keyword_end'], 'new_keyword_end': updated['keyword_end']})
        if {k: v for k, v in row.items() if k != 'keyword_end'} != {k: v for k, v in updated.items() if k != 'keyword_end'}:
            raise ValueError('Non-endpoint feature metadata changed')
        path = Path(row['path'])
        path = path if path.is_absolute() else base / path
        if sha(path) != row['features_sha256']:
            raise ValueError('Original feature payload differs')
        rows.append(updated)
        output_jobs.append(updated_job)
    if not changed:
        raise ValueError('No eligible late endpoint; do not fit an empty intervention')
    args.output.mkdir(parents=True)
    (args.output / 'index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (args.output / 'jobs-private.json').write_text(json.dumps(output_jobs) + '\n')
    identity = json.loads((reference / 'identity.json').read_text())
    identity['metadata_endpoint_override'] = {'plan_sha256': sha(args.plan),
        'reference_index_sha256': sha(reference / 'index.jsonl'), 'script_sha256': sha(Path(__file__))}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'jobs': len(rows),
        'all_training_only': True, 'pilot_only': False, 'audio_speed_changed': False,
        'all_original_feature_payloads_verified': True, 'unchanged_feature_arrays': len(rows),
        'changed_endpoint_rows': len(changed), 'changed_unique_audio_sources': len(replacements),
        'changed_rows_by_engine': dict(sorted(Counter(r['engine'] for r in changed).items())),
        'other_row_fields_and_order_unchanged': True, 'heldout_sources_disjoint': True,
        'base_feature_indexes_untouched': True, 'representative_input_rule_unchanged': True,
        'accepted': False}
    (args.output / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'endpoint-audit.json').write_text(json.dumps({**report,
        'changed_rows': changed, 'changed_sources': list(replacements.values())}, indent=2) + '\n')
    print(json.dumps({**report, 'index_sha256': sha(args.output / 'index.jsonl'),
        'complete_sha256': sha(args.output / 'complete.json')}), flush=True)


if __name__ == '__main__':
    main()
