"""Inventory exact PCM support of old training-only isolated wake/confuser WAVs.

All original PCM is preserved. Nonzero bounds are an exact digital-support
description, not an estimated phonetic endpoint or a new label. No detector,
feature extractor, sample playback or model training is invoked.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import wave

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fits_support(first, end, rir_tail, offsets, context=150):
    """Existence under the GPU full-signal sample ledger, before raw-array checks.

    `end` is exclusive. Source is placed after32000 zero samples; earliest
    score must contain the end, latest window must still contain the start.
    This copies the conservative GPU sample-bound convention; it does not
    claim that a feature pack or model has actually been constructed/validated.
    """
    if first is None or end is None:
        return False
    score_end = (32000+end+rir_tail+159)//160-1
    return ((score_end+offsets-context)*160 <= 32000+first
            and (score_end+1)*160 >= 32000+end+rir_tail)


def read_support(path, record):
    if sha(path.read_bytes()) != record['sha256']:
        raise ValueError('Original WAV hash mismatch')
    with wave.open(str(path), 'rb') as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (1, 2, 16000, record['samples']):
            raise ValueError('Original WAV header mismatch')
        data = stream.readframes(stream.getnframes())
    if len(data) != record['samples']*2:
        raise ValueError('Truncated PCM')
    pcm = np.frombuffer(data, dtype='<i2')
    indices = np.flatnonzero(pcm)
    first, end = (int(indices[0]), int(indices[-1])+1) if len(indices) else (None, None)
    return first, end, sha(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source-root', 'plan', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_bytes())
    manifest_bytes = args.manifest.read_bytes()
    assert sha(manifest_bytes) == plan['manifest_sha256']
    records = [json.loads(line) for line in manifest_bytes.splitlines()]
    selected = []
    for row in records:
        text = ' '.join(re.findall('[a-z]+', row['text'].lower()))
        if row['split'] != 'training' or text not in plan['phrase_labels']:
            continue
        assert row['label'] == plan['phrase_labels'][text]
        selected.append((row, text))
    assert len(selected) == plan['selected_rows'] == 3393
    root = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary, details = {}, []
    for row, phrase in selected:
        member = PurePosixPath(row['path'])
        if member.is_absolute() or '..' in member.parts or member.parts[0] != 'workspace':
            raise ValueError('Unsafe original source member')
        source = (root/Path(*member.parts[1:])).resolve()
        source.relative_to(root)
        first, end, pcm_sha = read_support(source, row)
        detail = {key: row[key] for key in ('path', 'engine', 'label', 'speakers', 'samples', 'sha256')}
        detail.update(phrase=phrase, pcm_sha256=pcm_sha, first_nonzero=first,
            last_nonzero_exclusive=end, all_zero=first is None,
            leading_zero_samples=first if first is not None else row['samples'],
            trailing_zero_samples=row['samples']-end if end is not None else row['samples'],
            support_geometry={})
        for tail in plan['rir_tail_samples']:
            for offsets in plan['last_window_offset_frames']:
                name = f'context150_offset{offsets}_rir{tail}'
                full = fits_support(0, row['samples'], tail, offsets)
                nonzero = fits_support(first, end, tail, offsets)
                detail['support_geometry'][name] = {'full_container': full, 'all_nonzero_pcm': nonzero}
        details.append(detail)
    for engine in sorted({r['engine'] for r in details}):
        summary[engine] = {}
        for label in (0, 1):
            group = [r for r in details if r['engine'] == engine and r['label'] == label]
            summary[engine][str(label)] = {'rows': len(group),
                'unique_source_sha256': len({r['sha256'] for r in group}),
                'voice_tokens': len({str(v) for r in group for v in r['speakers']}),
                'all_zero_rows': sum(r['all_zero'] for r in group), 'geometry': {}}
            for name in details[0]['support_geometry']:
                summary[engine][str(label)]['geometry'][name] = {
                    'full_container_fits': sum(r['support_geometry'][name]['full_container'] for r in group),
                    'all_nonzero_pcm_fits': sum(r['support_geometry'][name]['all_nonzero_pcm'] for r in group),
                    'new_rows_with_only_exact_zero_padding_excluded': sum(
                        r['support_geometry'][name]['all_nonzero_pcm'] and not r['support_geometry'][name]['full_container'] for r in group)}
    content = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in details).encode()
    (args.output/'sources.jsonl').write_bytes(content)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan.read_bytes()),
        'source_sha256': sha(Path(__file__).read_bytes()), 'manifest_sha256': sha(manifest_bytes),
        'sources_jsonl_sha256': sha(content), 'selected_rows': len(details),
        'summary': summary, 'all_original_wav_hashes_headers_verified': True,
        'waveform_changed': False, 'detector_scored': False, 'model_trained': False,
        'limits': plan['limits']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    key = 'context150_offset12_rir960'
    print(json.dumps({'rows': len(details), 'five_score_rir_geometry': {
        e: {label: g['geometry'][key] for label, g in groups.items()} for e, groups in summary.items()}}))


if __name__ == '__main__':
    main()
