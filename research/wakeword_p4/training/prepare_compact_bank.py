"""Retain every eligible training-only same-voice phrase pair, without a teacher.

This is a fresh bank for a separate experiment. Original source labels,
render variants and whole-phrase geometry are preserved. Repeated crops are
counted explicitly; pair count is not an independent utterance count.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from near_word_windows import load_windows
from paired_training import bank, voice_key
from phrase_ctc_readout import phrase_class, sha

FRAMES = 335


def select_pairs(rows, bounds):
    """Full Cartesian partners within each original fixed-voice/render group."""
    keys = [(row['path'], row['variant']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Ambiguous source/render identity')
    result = []
    def identity(item):
        row, _ = item
        return row['path'], row['variant'], row['sha256']
    for stratum, entries in bank(rows, bounds, FRAMES).items():
        for negative, positives in sorted(entries, key=lambda item: identity(item[0])):
            for positive in sorted(positives, key=identity):
                if (voice_key(positive[0]) != voice_key(negative[0])
                        or phrase_class(positive[0]) != 0 or phrase_class(negative[0]) == 0):
                    raise ValueError('Mismatched voice/render or phrase role')
                result.append((stratum, (positive, negative)))
    return result


def coverage(pairs):
    rows = [row for _, items in pairs for row, _ in items]
    by_stratum = Counter(':'.join(stratum) for stratum, _ in pairs)
    crops = {(row['path'], row['variant'], sum(band)//2)
             for _, items in pairs for row, band in items}
    return {'pairs': len(pairs), 'rows': len(rows), 'unique_crops': len(crops),
            'unique_sources': len({row['sha256'] for row in rows}),
            'positive_sources': len({row['sha256'] for row in rows if row['label']}),
            'negative_sources': len({row['sha256'] for row in rows if not row['label']}),
            'pairs_by_stratum': dict(sorted(by_stratum.items())),
            'engines': sorted({row['engine'] for row in rows})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'features', 'windows', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    plan = json.loads(args.plan.read_bytes())
    source = Path(__file__).parent
    if any(sha(source/name) != digest for name, digest in plan['source_sha256'].items()):
        raise ValueError('Source differs from the frozen bank plan')
    index = (args.features/'features.json').read_bytes()
    if (hashlib.sha256(index).hexdigest() != plan['feature_index_sha256']
            or sha(args.windows) != plan['windows_sha256']):
        raise ValueError('Feature-index/window identity differs')
    rows = json.loads(index)
    bounds, _ = load_windows(args.windows, index, rows, FRAMES)
    pairs = select_pairs(rows, bounds)
    measured = coverage(pairs)
    if measured != plan['expected_coverage']:
        raise ValueError('Source/render/pair coverage differs')
    # Read-only hashing binds the actual feature bytes, not just their index.
    feature_sha = sha(args.features/'features.bin')
    if feature_sha != plan['feature_data_sha256']:
        raise ValueError('Full feature-cache identity differs')
    raw = np.memmap(args.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    samples = np.empty((len(pairs)*2, FRAMES, 40), np.uint16)
    retained = []
    for pair, (stratum, items) in enumerate(pairs):
        for role, (row, band) in enumerate(items):
            end = sum(band)//2
            at = pair*2+role
            sample = raw[row['offset']+end-FRAMES:row['offset']+end]
            if sample.shape != (FRAMES, 40):
                raise ValueError('Incomplete cached feature crop')
            samples[at] = sample
            retained.append({'stratum': list(stratum), 'pair': pair, 'label': row['label'],
                'path': row['path'], 'source_sha256': row['sha256'], 'variant': row['variant'],
                'end_frame': end, 'bounds': list(band),
                'feature_sha256': hashlib.sha256(sample.tobytes()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'bank.npz', features=samples)
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'coverage': measured,
        'feature_index_sha256': sha(args.features/'features.json'),
        'feature_data_sha256': feature_sha, 'windows_sha256': sha(args.windows),
        'source_files_sha256': plan['source_sha256'], 'frames': FRAMES,
        'pair_count': len(pairs), 'data_sha256': sha(args.output/'bank.npz'), 'rows': retained,
        'sampling': 'All same-voice/render Cartesian pairs at complete-band midpoint; later pair sampling is uniform over these pairs, not engines.',
        'limits': plan['limits']}
    (args.output/'bank.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'coverage': measured, 'bank_sha256': sha(args.output/'bank.json'),
        'bank_data_sha256': report['data_sha256'], 'feature_data_sha256': feature_sha}), flush=True)


if __name__ == '__main__':
    main()
