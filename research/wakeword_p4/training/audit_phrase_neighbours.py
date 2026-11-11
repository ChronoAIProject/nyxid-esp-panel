"""Training-only acoustic neighbours for blind pronunciation review.

Distances are geometry diagnostics, not phonetic truth or wake scores. Source
PCM is only read at its original rate. DTW aligns feature indices to compute a
distance; it never stretches, synthesizes, plays or changes any audio.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np

from paired_training import voice_key
from phrase_ctc_readout import phrase_class, sha


def dtw_distance(left, right):
    """Minimum cosine path cost, normalized by mean input length.

    Center/normalize each spectral frame without altering its time axis.
    Use all three standard monotone DTW moves. The fixed normalization is
    symmetric and independent of path tie-breaking; this is not min-mean DTW.
    """
    arrays = [np.asarray(a, np.float64) for a in (left, right)]
    if any(a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Expected finite nonempty frame sequences')
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError('Spectral dimensions differ')
    units, nonzero = [], []
    for value in arrays:
        centered = value-value.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        units.append(np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-12))
        nonzero.append(norm[:, 0] > 1e-12)
    cost = np.clip(1-units[0]@units[1].T, 0, 2)
    cost[np.ix_(~nonzero[0], ~nonzero[1])] = 0
    previous = np.full(len(right)+1, np.inf); previous[0] = 0
    for row in cost:
        current = np.full(len(right)+1, np.inf)
        for j, value in enumerate(row, 1):
            current[j] = value+min(previous[j], current[j-1], previous[j-1])
        previous = current
    return float(previous[-1]*2/(len(left)+len(right)))


def source_bounds(pcm):
    """Fixed RMS activity proxy, never a phone boundary or pronunciation label."""
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError('Expected a nonempty mono source')
    blocks = np.pad(pcm, (0, -len(pcm) % 160)).reshape(-1, 160)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    active = np.flatnonzero(rms > max(1e-6, float(rms.max())*.005))
    if not len(active):
        raise ValueError('No acoustic activity')
    return int(active[0]), int(active[-1])+1


def training_rows(index, bank):
    """Only matched-bank variant0 sources; reject reserved waveform overlap."""
    lookup = {(r['path'], r['variant']): r for r in index}
    if len(lookup) != len(index):
        raise ValueError('Ambiguous source/render index')
    reserved = {r['sha256'] for r in index if r['split'] != 'training'}
    paths = sorted({r['path'] for r in bank['rows'] if r['variant'] == 0})
    selected = []
    for path in paths:
        rel = Path(path)
        row = lookup[path, 0]
        if (rel.is_absolute() or '..' in rel.parts or row['split'] != 'training'
                or row['sha256'] in reserved or voice_key(row) is None):
            raise ValueError('Invalid or reserved source identity')
        role = phrase_class(row)
        if any(r['source_sha256'] != row['sha256'] or r['label'] != row['label']
               for r in bank['rows'] if r['path'] == path):
            raise ValueError('Bank/source identity differs')
        selected.append({**row, 'class': role})
    if not selected:
        raise ValueError('No eligible training source')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'features', 'bank', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan['source_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, ('Source differs', name)
    for path, expected in ((args.features/'features.json', plan['feature_index_sha256']),
            (args.features/'features.bin', plan['feature_data_sha256']),
            (args.bank/'bank.json', plan['bank_sha256'])):
        assert sha(path) == expected, ('Input differs', path)
    index = json.loads((args.features/'features.json').read_bytes())
    bank = json.loads((args.bank/'bank.json').read_bytes())
    rows = training_rows(index, bank)
    assert len(rows) == plan['expected_source_paths']
    assert len({r['sha256'] for r in rows}) == plan['expected_source_hashes']
    data = np.memmap(args.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    groups, inventory, retained_features = defaultdict(dict), [], []
    for number, row in enumerate(rows):
        path = args.features/'dataset'/row['path']
        assert sha(path) == row['sha256'], ('Source WAV differs', row['path'])
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, row['samples'])
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float64)/32768
        first, end = source_bounds(pcm)
        # Cached 30ms overlapping frontend windows and conservative guards.
        lo, hi = max(0, row['lead_frame']+first-4), min(row['frames'], row['lead_frame']+end+6)
        assert hi > lo
        value = data[row['offset']+lo:row['offset']+hi].copy()
        group, digest = voice_key(row), row['sha256']
        item = {'id': number, 'engine': row['engine'], 'class': row['class'],
            'path': row['path'], 'source_sha256': digest, 'text': row['text'],
            'voice_group': list(group), 'activity_proxy_bins': [first, end],
            'feature_bounds': [lo, hi], 'feature_sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        inventory.append(item)
        retained_features.append(value)
        if digest in groups[group]:
            retained = groups[group][digest]
            assert retained['metadata']['class'] == row['class'], 'Identical PCM has conflicting intents'
            retained['aliases'].append(number)
        else:
            groups[group][digest] = {'metadata': item, 'features': value.astype(np.float64)/25.6, 'aliases': [number]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source').mkdir()
    for name in plan['source_sha256']:
        shutil.copyfile(Path(__file__).parent/name, args.output/'source'/name)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    (args.output/'launch.json').write_text(json.dumps({'checkout_revision': revision,
        'command': [sys.executable, *sys.argv], 'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'numpy': np.__version__}, indent=2)+'\n')
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    (args.output/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    np.savez_compressed(args.output/'features.npz', **{f'source_{i}': value for i, value in enumerate(retained_features)})
    comparisons, group_reports, distances = [], [], []
    for group in sorted(groups, key=lambda group: json.dumps(group)):
        members = list(groups[group].values())
        positives = [i for i, value in enumerate(members) if value['metadata']['class'] == 0]
        negatives = [i for i, value in enumerate(members) if value['metadata']['class'] != 0]
        matrix = np.full((len(members), len(members)), np.nan)
        np.fill_diagonal(matrix, 0)
        # Only distances needed by the predeclared comparison; no scores pick inputs.
        for i in positives:
            for j in range(len(members)):
                if not np.isfinite(matrix[i, j]):
                    matrix[i, j] = matrix[j, i] = dtw_distance(members[i]['features'], members[j]['features'])
        group_reports.append({'voice_group': list(group),
            'source_ids': [v['metadata']['id'] for v in members],
            'aliases': [v['aliases'] for v in members],
            'positive_source_hashes': len(positives), 'negative_source_hashes': len(negatives),
            'control_available': len(positives) >= 2})
        distances.append(matrix)
        if len(positives) < 2:
            continue
        for i in positives:
            j = min((j for j in positives if j != i), key=lambda j: (matrix[i, j], members[j]['metadata']['id']))
            for label in (1, 2, 3):
                choices = [j for j in negatives if members[j]['metadata']['class'] == label]
                if not choices:
                    continue
                k = min(choices, key=lambda k: (matrix[i, k], members[k]['metadata']['id']))
                comparisons.append({'engine': members[i]['metadata']['engine'], 'class': label,
                    'anchor_id': members[i]['metadata']['id'], 'positive_id': members[j]['metadata']['id'],
                    'negative_id': members[k]['metadata']['id'], 'positive_distance': float(matrix[i, j]),
                    'negative_distance': float(matrix[i, k]), 'margin': float(matrix[i, k]-matrix[i, j])})
    np.savez_compressed(args.output/'distances.npz', **{f'group_{i}': m for i, m in enumerate(distances)})
    (args.output/'groups.json').write_text(json.dumps(group_reports, indent=2)+'\n')
    (args.output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    cohorts, review = [], []
    for engine, label in sorted({(r['engine'], r['class']) for r in comparisons}):
        values = [r for r in comparisons if (r['engine'], r['class']) == (engine, label)]
        margins = np.array([r['margin'] for r in values])
        cohorts.append({'engine': engine, 'class': label, 'anchors': len(values),
            'negative_closer_than_positive_reference': int(np.sum(margins < 0)),
            'ties': int(np.sum(margins == 0)), 'median_margin': float(np.median(margins))})
        review.extend(sorted(values, key=lambda r: (r['margin'], r['anchor_id'], r['negative_id']))[:plan['review_per_engine_class']])
    (args.output/'review-candidates.json').write_text(json.dumps(review, indent=2)+'\n')
    report = {'scope': plan['scope'], 'plan_sha256': sha(args.plan),
        'source_paths': len(rows), 'source_hashes': len({r['sha256'] for r in rows}),
        'voice_groups': len(groups), 'control_groups': sum(r['control_available'] for r in group_reports),
        'collapsed_alias_paths': sum(len(v['aliases'])-1 for g in groups.values() for v in g.values()),
        'computed_pair_distances': sum(int(np.isfinite(m[np.triu_indices(len(m), 1)]).sum()) for m in distances),
        'cohorts': cohorts, 'review_candidates': len(review), 'accepted': False,
        'evidence_sha256': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
        'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
