"""Build model-free complete-container features from the frozen old inventory.

No original WAV is trimmed, resampled, relabeled or overwritten. Exact WAV
aliases share one canonical source and retain every original metadata row.
This constructs candidate input data, without a sampler or training recipe.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import wave

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def group_sources(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['support_geometry']['context150_offset12_rir960']['full_container']:
            groups[row['sha256']].append(row)
    ordered = []
    for digest, aliases in groups.items():
        aliases.sort(key=lambda r: r['path'])
        if len({(r['engine'], r['label'], r['samples']) for r in aliases}) != 1:
            raise ValueError('Conflicting exact source aliases')
        ordered.append({'source_sha256': digest, 'aliases': aliases})
    return sorted(ordered, key=lambda g: g['aliases'][0]['path'])


def render(pcm, path, variant):
    """Match the existing British pack's three fixed rendering conditions."""
    from scipy.signal import fftconvolve
    if variant not in (0, 1, 2):
        raise ValueError('Unsupported rendering variant')
    seed = int(sha((path+f':{variant}:phone-v1').encode())[:16], 16)
    rng = np.random.default_rng(seed)
    ir = np.zeros(961, dtype=np.float64)
    ir[[0, 160, 480, 960]] = [1, .25, -.12, .06]
    speech = fftconvolve(pcm, ir, mode='full') if variant == 2 else pcm.copy()
    signal = speech*(.08/max(1e-8, float(np.sqrt(np.mean(speech**2)))))
    stream = np.pad(signal, (32000, 16000))
    snr = None
    if variant:
        snr = 25 if variant == 1 else 15
        noise = rng.normal(0, 1, len(stream))
        noise *= .08*10**(-snr/20)/np.sqrt(np.mean(noise**2))
        stream += noise
    gain = min(1., .95/max(float(np.max(np.abs(stream))), 1e-8))
    stream *= gain
    if not np.isfinite(stream).all() or np.max(np.abs(stream)) >= 1:
        raise ValueError('Invalid rendered stream')
    return stream, {'seed': seed, 'snr_db': snr, 'gain_after_mix': gain,
                    'signal_samples': len(signal), 'rir_tail_samples': 960 if variant == 2 else 0}


def source_path(root, member):
    member = PurePosixPath(member)
    if member.is_absolute() or '..' in member.parts or member.parts[0] != 'workspace':
        raise ValueError('Unsafe source path')
    result = (root/Path(*member.parts[1:])).resolve()
    result.relative_to(root)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'source-root', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    inventory_bytes = args.inventory.read_bytes()
    if sha(inventory_bytes) != plan['inventory_sha256']:
        raise ValueError('Inventory differs from frozen plan')
    rows = [json.loads(line) for line in inventory_bytes.splitlines()]
    groups = group_sources(rows)
    if (len(rows), len(groups), sum(len(g['aliases']) for g in groups)) != (3393, 2594, 2632):
        raise ValueError('Frozen source/alias counts differ')
    frontend_path = Path(__file__).with_name('features.py')
    if sha(frontend_path.read_bytes()) != plan['frontend_source_sha256']:
        raise ValueError('Frontend source differs')
    from features import features
    import importlib.metadata
    import scipy
    if importlib.metadata.version('pymicro-features') != plan['pymicro_features_version']:
        raise ValueError('Frontend package differs')
    if np.__version__ != plan['numpy_version'] or scipy.__version__ != plan['scipy_version']:
        raise ValueError('Numerical package version differs')
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.source_root.resolve()
    raw_dir = args.output/'raw'
    raw_dir.mkdir()
    windows, sequences, labels, records = [], [], [], []
    started = time.monotonic()
    for source_index, group in enumerate(groups):
        row = group['aliases'][0]
        path = source_path(root, row['path'])
        if sha(path.read_bytes()) != group['source_sha256']:
            raise ValueError('Canonical original WAV differs')
        with wave.open(str(path), 'rb') as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes(), stream.getcomptype()) != (1, 2, 16000, row['samples'], 'NONE'):
                raise ValueError('Canonical original WAV header differs')
            pcm_bytes = stream.readframes(stream.getnframes())
        if len(pcm_bytes) != 2*row['samples'] or sha(pcm_bytes) != row['pcm_sha256']:
            raise ValueError('Canonical original PCM differs')
        pcm = np.frombuffer(pcm_bytes, dtype='<i2').astype(np.float64)/32768
        for variant in plan['variants']:
            signal, rendered = render(pcm, row['path'], variant)
            # Preserve the existing conservative sample ledger and frame phase.
            end = (32000+rendered['signal_samples']+159)//160-1
            input_pcm = np.clip(np.rint(signal*32768), -32768, 32767).astype(np.int16)
            raw = features(input_pcm)
            if raw.shape != ((len(input_pcm)+159)//160-2, 40) or raw.dtype != np.uint16:
                raise ValueError('Full-history raw feature shape/type differs')
            bounds = [((end+d-150)*160, (end+d+1)*160) for d in (0, 3, 6, 9, 12)]
            if not all(begin <= 32000 and stop >= 32000+rendered['signal_samples'] for begin, stop in bounds):
                raise ValueError('A window fails whole-source/RIR coverage')
            seq = raw[end-150:end+12]
            if end < 150 or end+12 > len(raw) or seq.shape != (162, 40):
                raise ValueError('Sequence crop is outside actual full-history array')
            raw_member = f'raw/{source_index:04d}-{variant}.npy'
            np.save(args.output/raw_member, raw, allow_pickle=False)
            windows.append(seq[:150]);sequences.append(seq);labels.append(row['label'])
            records.append({'pack_index': len(records), 'source_index': source_index,
                'source_group': row['engine'], 'source_sha256': group['source_sha256'],
                'canonical_path': row['path'], 'label': row['label'], 'variant': variant,
                'source_samples': row['samples'], **rendered,
                'feature_crop_end': end, 'five_window_sample_bounds': bounds,
                'rendered_pcm_sha256': sha(input_pcm.tobytes()),
                'raw': raw_member, 'raw_shape': list(raw.shape),
                'raw_sha256': sha(raw.tobytes()),
                'sequence_sha256': sha(seq.tobytes()), 'window_sha256': sha(seq[:150].tobytes())})
        if source_index % 100 == 0:
            print(json.dumps({'sources_done': source_index+1, 'total': len(groups)}), flush=True)
    pack = args.output/'training.npz'
    np.savez_compressed(pack, x=np.asarray(windows, dtype=np.uint16),
                        sequence_x=np.asarray(sequences, dtype=np.uint16), y=np.asarray(labels, dtype=np.float32))
    (args.output/'sources.json').write_text(json.dumps(groups, indent=2)+'\n')
    (args.output/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    report = {'role': 'candidate-data-only-complete-original-container-pack',
        'plan_sha256': sha(plan_bytes), 'inventory_sha256': sha(inventory_bytes),
        'script_sha256': sha(Path(__file__).read_bytes()),
        'frontend_source_sha256': sha(frontend_path.read_bytes()),
        'original_alias_rows': 2632, 'unique_sources': len(groups), 'variants': len(records),
        'excluded_inventory_rows': 3393-2632, 'exact_zero_exclusion_used': False,
        'pack_sha256': sha(pack.read_bytes()),
        'sources_sha256': sha((args.output/'sources.json').read_bytes()),
        'records_sha256': sha((args.output/'records.json').read_bytes()),
        'counts': dict(Counter(f"{r['source_group']}/label{r['label']}/variant{r['variant']}" for r in records)),
        'all_five_sample_bounds_pass': True, 'all_crops_within_real_history': True,
        'elapsed_seconds': time.monotonic()-started,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'pymicro_features': importlib.metadata.version('pymicro-features'), 'platform': platform.platform()},
        'training_run': False, 'model_scored': False, 'limits': plan['limits']}
    (args.output/'plan.json').write_bytes(plan_bytes)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
