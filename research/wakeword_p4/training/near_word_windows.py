"""Prepare whole-phrase training windows for standalone near words."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

NEAR_WORDS = frozenset(('hey max', 'hey mix', 'hey next', 'hey nexus', 'hey nice', 'hey nick', 'hey six'))


def eligible(row):
    return (row['split'] == 'training' and row['label'] == 0
            and row['text'].lower().strip().rstrip('.!?') in NEAR_WORDS)


def key(row):
    return row['path'], row['variant']


def load_windows(file, feature_index_bytes, rows, frames):
    raw = file.read_bytes()
    data = json.loads(raw)
    if (data['feature_index_sha256'] != hashlib.sha256(feature_index_bytes).hexdigest()
            or data['context_frames'] != frames):
        raise ValueError('Near-word windows do not bind this feature index and context')
    reserved = {r['sha256'] for r in rows if r['split'] != 'training'}
    expected = {key(row): row for row in rows if eligible(row)}
    windows = {}
    for item in data['windows']:
        k = key(item)
        row = expected[k]
        if k in windows or row['sha256'] in reserved or item['sha256'] != row['sha256']:
            raise ValueError('Invalid source role, identity, or bounds for near-word window')
        bounds = item['bounds']
        if bounds is not None:
            low, high = bounds
            if (not all(type(n) is int for n in bounds)
                    or not max(frames, row['lead_frame'] - 25) <= low <= high <= row['frames']
                    or high - frames > row['lead_frame'] - 1):
                raise ValueError('Window exceeds complete source geometry')
        windows[k] = bounds
    if not windows or windows.keys() != expected.keys():
        raise ValueError('Incomplete near-word windows')
    return windows, hashlib.sha256(raw).hexdigest()


def map_end(end, lo, hi, bounds):
    left, right = bounds
    return left + (end - lo) * (right - left + 1) // (hi - lo + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.frames < 1:
        parser.error('Specify positive context and a fresh output path')
    index = (args.features / 'features.json').read_bytes()
    rows = json.loads(index)
    reserved = {r['sha256'] for r in rows if r['split'] != 'training'}
    groups = defaultdict(list)
    for row in rows:
        if eligible(row):
            groups[row['path']].append(row)
    if not groups:
        raise ValueError('No near-word training sources')
    windows, counts = [], Counter()
    original_complete_fraction = []
    for path, variants in sorted(groups.items()):
        row = variants[0]
        rel = Path(path)
        if rel.is_absolute() or '..' in rel.parts or row['sha256'] in reserved:
            raise ValueError('Path or source SHA overlaps a reserved split')
        wav = args.features / 'dataset' / rel
        if hashlib.sha256(wav.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Original WAV checksum mismatch')
        pcm, rate = sf.read(wav, dtype='float64')
        if rate != 16000 or pcm.shape != (row['samples'],) or sf.info(wav).subtype != 'PCM_16':
            raise ValueError('Unexpected original PCM geometry')
        block = np.pad(pcm, (0, -len(pcm) % 160)).reshape(-1, 160)
        rms = np.sqrt(np.mean(block ** 2, axis=1))
        active = np.flatnonzero(rms > max(1e-6, float(rms.max()) * .005))
        if not len(active):
            raise ValueError('No acoustic activity in a labelled near word')
        # Use the last active 10ms bin plus 60ms, matching the existing
        # non-Piper positive anchor convention. Never crop the underlying PCM.
        end = min(len(pcm), (int(active[-1]) + 1) * 160 + 960)
        counts[row['engine']] += 1
        if sorted(v['variant'] for v in variants) != list(range(4)):
            raise ValueError('Expected four cached renders per source')
        for v in variants:
            if (v['sha256'], v['engine'], v['text'], v['samples']) != (
                    row['sha256'], row['engine'], row['text'], row['samples']):
                raise ValueError('Variant/source metadata differs')
            # Frames overlap by 20ms. The lead index is floored, so allow
            # another 10ms when verifying whole-source history coverage.
            endpoint = v['lead_frame'] + (end + 159) // 160 - 2
            orig_lo = max(args.frames, v['lead_frame'] - 25)
            low = max(orig_lo, endpoint + 3)
            high = min(v['frames'], endpoint + 27, v['lead_frame'] - 1 + args.frames)
            # A long source may not fit. Retain its original random draws,
            # with the excluded intervention accounted for explicitly.
            bounds = [low, high] if low <= high else None
            if bounds:
                original_complete_fraction.append((high - low + 1) / (v['frames'] - orig_lo + 1))
            windows.append({'path': path, 'variant': v['variant'], 'sha256': row['sha256'],
                            'bounds': bounds})
    if not original_complete_fraction:
        raise ValueError('No complete source windows fit this context')
    data = {'feature_index_sha256': hashlib.sha256(index).hexdigest(), 'context_frames': args.frames,
            'sources': len(groups), 'views': len(windows), 'sources_by_engine': counts,
            'views_unchanged_due_to_source_length': len(windows) - len(original_complete_fraction),
            'unweighted_mean_uniform_draw_fraction_inside_scheduled_band': float(np.mean(original_complete_fraction)),
            'windows': windows}
    with args.output.open('x') as handle:
        handle.write(json.dumps(data, separators=(',', ':')) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'windows'}), flush=True)


if __name__ == '__main__':
    main()
