"""Make immutable training-only feature views for a consistency-screen ablation.

Neither ASR nor phoneme CTC is pronunciation ground truth. Preserve every
source label, waveform and embedding; exclude uncertain clips from a separate
view and compare it with the complete view on unchanged validation.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import re

import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical_phones(phones):
    phones = phones[:12]
    for start in range(len(phones)-4):
        first, n, vowel, k, s = phones[start:start+5]
        if first in {'eɪ', 'eː', 'e', 'ɛ'} and n == 'n' and vowel in {'ɪ', 'i', 'iː'} and k == 'k' and s == 's':
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--phonemes', type=Path, required=True)
    parser.add_argument('--asr', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--features', type=Path, action='append', required=True)
    parser.add_argument('--name', action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if len(args.features) != len(args.name) or len(set(args.name)) != len(args.name):
        parser.error('Supply one unique --name per --features directory')
    for name in args.name:
        if Path(name).parts != (name,) or name in ('.', '..') or (args.output_root/name).exists():
            parser.error('Output names must be new children of output-root')
    if args.report.exists():
        parser.error('Do not overwrite an existing report')
    plan = json.loads(args.plan.read_text())
    policy = plan['consistency_screen']
    canonical = {'hey nicks', 'hey nix', 'hey nyx', 'hey knicks'}
    if set(policy['canonical_asr']) != canonical:
        raise ValueError('This implementation requires the frozen canonical transcript set')
    if (policy['positive_keep'] != 'Canonical phoneme pattern OR canonical unprompted ASR transcript' or
            policy['negative_keep'] != 'Neither canonical phoneme pattern nor canonical ASR transcript' or
            not policy['phoneme_rule'].startswith('Within first12greedy phones: optional h, eɪ/eː/e/ɛ, n, ɪ/i/iː, k, s;')):
        raise ValueError('This implementation requires the frozen consistency predicate')
    source_index = args.source/'manifest.jsonl'
    sources = [json.loads(line) for line in source_index.read_text().splitlines()]
    phonemes = [json.loads(line) for line in args.phonemes.read_text().splitlines()]
    asr = [json.loads(line) for line in args.asr.read_text().splitlines()]
    by_phone = {r['path']: r for r in phonemes}
    by_asr = {r['path']: r for r in asr}
    if len(by_phone) != len(phonemes) or len(by_asr) != len(asr):
        raise ValueError('Duplicate consistency-audit rows')
    paths = {r['path'] for r in sources}
    if len(paths) != len(sources) or paths != set(by_phone) or paths != set(by_asr):
        raise ValueError('Require complete source-matched audits before screening')
    keep, reasons = set(), Counter()
    by_hash = {}
    for source in sources:
        path = Path(source['path'])
        if source['split'] != 'training' or path.is_absolute() or '..' in path.parts:
            raise ValueError('Only source-relative training clips may be screened')
        if digest(args.source/path) != source['sha256']:
            raise ValueError('Source WAV checksum mismatch')
        phone, transcript = by_phone[source['path']], by_asr[source['path']]
        if phone['sha256'] != source['sha256'] or transcript['sha256'] != source['sha256']:
            raise ValueError('Audit/source identity mismatch')
        if phone['source_label'] != source['label'] or transcript['label'] != source['label']:
            raise ValueError('Audit/source label mismatch')
        if source['sha256'] in by_hash:
            raise ValueError('Duplicate waveform identity requires a separate audit')
        by_hash[source['sha256']] = source
        words = ' '.join(re.findall(r'[a-z]+', transcript['transcript'].lower()))
        sounds_canonical = canonical_phones(phone['phonemes']) or words in canonical
        if bool(source['label']) == sounds_canonical:
            keep.add(source['sha256'])
        else:
            reasons['positive_without_consistency' if source['label'] else 'ambiguous_negative'] += 1
    outputs = []
    for directory, name in zip(args.features, args.name):
        rows = [json.loads(line) for line in (directory/'index.jsonl').read_text().splitlines()]
        complete = json.loads((directory/'complete.json').read_text())
        jobs = json.loads((directory/'jobs-private.json').read_text())
        if len(rows) != len(jobs) or len(rows) != complete['jobs'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Incomplete or duplicate feature source')
        selected = []
        for row in rows:
            if row['split'] != 'training' or row['source_sha256'] not in by_hash:
                raise ValueError('Feature row is outside the audited training source')
            if row['label'] != by_hash[row['source_sha256']]['label']:
                raise ValueError('Feature/source label mismatch')
            path = Path(row['path'])
            if not path.is_absolute():
                raise ValueError('This view requires absolute immutable feature paths')
            if digest(path) != row['features_sha256']:
                raise ValueError('Feature checksum differs')
            array = np.load(path)
            if array.dtype != np.int8 or array.shape != (row['frames'], 96):
                raise ValueError('Feature shape/type differs')
            if row['source_sha256'] in keep:
                selected.append(row)
        ids = {r['id'] for r in selected}
        selected_jobs = [job for job in jobs if job['id'] in ids]
        if len(selected_jobs) != len(selected) or not any(r['label'] for r in selected):
            raise ValueError('Empty positive view or inconsistent jobs')
        output = args.output_root/name
        output.mkdir()
        identity = json.loads((directory/'identity.json').read_text())
        identity.update({'scope': 'Training-only consistency-screen view; original sources/labels/features preserved',
                         'source_index_sha256': digest(directory/'index.jsonl'), 'plan_sha256': digest(args.plan),
                         'screen_script_sha256': digest(Path(__file__)), 'jobs': len(selected)})
        (output/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
        (output/'jobs-private.json').write_text(json.dumps(selected_jobs)+'\n')
        (output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
        (output/'complete.json').write_text(json.dumps({'jobs': len(selected), 'source_jobs': len(rows),
                                                      'training_only': True}, indent=2)+'\n')
        outputs.append({'directory': name, 'index_sha256': digest(output/'index.jsonl'),
                        'complete_sha256': digest(output/'complete.json'), 'views': len(selected)})
    report = {'scope': 'Imperfect training-only consistency ablation; not pronunciation or recognition acceptance',
              'source_clips': len(sources), 'retained_source_clips': len(keep),
              'retained_positive_sources': sum(by_hash[h]['label'] for h in keep),
              'excluded_by_reason': dict(reasons), 'source_manifest_sha256': digest(source_index),
              'phonemes_sha256': digest(args.phonemes), 'asr_sha256': digest(args.asr),
              'plan_sha256': digest(args.plan), 'script_sha256': digest(Path(__file__)),
              'original_paths_hashes_labels_unchanged': True, 'outputs': outputs, 'accepted': False}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
