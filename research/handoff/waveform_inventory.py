"""Verify portable, original synthetic WAVs for a different feature frontend.

Uses only the exported base training/validation jobs. Drops derived frame
coordinates: keyword_end is an approximate source-PCM sample index, not a
10 ms MicroFrontend or 80 ms embedding frame. No model scores are read.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import wave


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_member(source, original_prefix):
    relative = PurePosixPath(source).relative_to(PurePosixPath(original_prefix))
    if '..' in relative.parts or relative.is_absolute():
        raise ValueError('Source escapes the declared original prefix')
    return str(PurePosixPath('workspace') / relative)


def validate_partitions(records):
    hashes, voices = defaultdict(set), defaultdict(set)
    for row in records:
        if row['split'] not in ('training', 'validation'):
            raise ValueError('Only existing training/validation may be inventoried')
        hashes[row['sha256']].add(row['split'])
        for speaker in row['speakers']:
            voices[(row['engine'], str(speaker))].add(row['split'])
    if any(len(splits) != 1 for splits in hashes.values()):
        raise ValueError('Source WAV hash crosses training/validation')
    if any(len(splits) != 1 for splits in voices.values()):
        raise ValueError('Source voice token crosses training/validation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--original-prefix', default='/Users/auric/tmp/output')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    snapshot = args.snapshot.resolve()
    contents = {}
    for name in ('embedding-features', 'synthetic-training-audio'):
        for member, expected in json.loads((snapshot / f'CONTENTS-{name}.json').read_text()).items():
            if member in contents and contents[member] != expected:
                raise ValueError('Conflicting downloaded source manifests')
            contents[member] = expected

    def verified(member):
        path = (snapshot / member).resolve()
        path.relative_to(snapshot)
        expected = contents[member]
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Downloaded source member differs: ' + member)
        return path

    prefix = 'workspace/hey-nyx-training-v3/'
    jobs_path = verified(prefix + 'embedding-feasibility/head-features-tts-v1/jobs-private.json')
    metadata_path = verified(prefix + 'features-expanded-human335/features.json')
    metadata = {}
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] != 0 or row['split'] not in ('training', 'validation'):
            continue
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid raw source metadata path')
        member = prefix + 'features-expanded-human335/dataset/' + str(relative)
        if member in metadata:
            raise ValueError('Duplicate base source metadata')
        metadata[member] = row

    records = []
    for job in json.loads(jobs_path.read_text()):
        if job['variant'] != 0:
            continue
        if job['kind'] != 'tts' or job['split'] not in ('training', 'validation'):
            raise ValueError('Unexpected source job')
        member = source_member(job['source'], args.original_prefix)
        row = metadata[member]
        if any(job[k] != row[k] for k in ('split', 'label', 'engine', 'keyword_end')):
            raise ValueError('Job and original source metadata differ')
        if job['source_sha256'] != row['sha256']:
            raise ValueError('Source hash differs between metadata and job')
        path = verified(member)
        if digest(path) != row['sha256']:
            raise ValueError('WAV differs from its original source metadata')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, row['samples']):
                raise ValueError('Original source is not the bound mono16k PCM16')
        end = row['keyword_end']
        if end is not None and not 0 < end <= row['samples']:
            raise ValueError('Keyword endpoint is outside the original PCM')
        records.append({k: row[k] for k in
                        ('split', 'label', 'engine', 'text', 'speakers', 'sha256', 'samples', 'keyword_end')}
                       | {'path': member, 'sample_rate': 16000})
    if {r['path'] for r in records} != set(metadata) or len(records) != len(metadata):
        raise ValueError('Jobs do not cover original metadata exactly once')
    records.sort(key=lambda r: r['path'])
    validate_partitions(records)
    args.output.mkdir(parents=True)
    manifest = args.output / 'waveforms.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    report = {'scope': __doc__, 'script_sha256': digest(Path(__file__)),
        'jobs_sha256': digest(jobs_path), 'source_metadata_sha256': digest(metadata_path),
        'manifest_sha256': digest(manifest), 'records': len(records),
        'counts': [{'engine': e, 'split': s, 'label': label, 'sources': n}
                   for (e, s, label), n in sorted(Counter(
                       (r['engine'], r['split'], r['label']) for r in records).items())],
        'source_wav_hashes_and_headers_verified': True,
        'cross_split_wav_hashes': 0, 'cross_split_voice_tokens': 0,
        'audio_changed': False, 'detector_scores_read': False,
        'limits': ['Existing synthetic development voices, not independent human speakers or a fresh holdout.',
                   'Original voice tokens are engine-qualified; Qwen conditions and Piper blends are not new human identities.',
                   'Preserve splits and group provenance when importing. New feature extraction/calibration is required.',
                   'No derived frame indices or precomputed feature arrays are exported.',
                   'Does not include the separate additional960Qwen sources, reserved final audio or microphone captures.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'scope'}), flush=True)


if __name__ == '__main__':
    main()
