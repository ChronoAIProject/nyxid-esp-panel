"""Read-only candidate-pack audit, independent of the pack builder.

Recompute source selection/aliases and every saved raw crop. Re-render only
the first canonical source per engine/label (all three conditions), using
SoundFile for source decoding and the same pinned public MicroFrontend.
No classifier, detector, sampler, training or audio device is invoked.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf


def sha(data):
    return hashlib.sha256(data).hexdigest()


def member(root, path):
    relative = PurePosixPath(path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe member')
    result = (root/Path(*relative.parts)).resolve()
    result.relative_to(root)
    return result


def reference_stream(pcm, path, variant):
    seed = int(hashlib.sha256((path+f':{variant}:phone-v1').encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    if variant == 2:
        impulse = np.zeros(961, dtype=np.float64)
        for delay, value in [(0, 1), (160, .25), (480, -.12), (960, .06)]:
            impulse[delay] = value
        speech = fftconvolve(pcm, impulse, mode='full')
    else:
        speech = pcm.copy()
    speech *= .08/max(1e-8, float(np.sqrt(np.mean(speech**2))))
    stream = np.pad(speech, (32000, 16000))
    if variant != 0:
        noise = rng.normal(0, 1, len(stream))
        noise *= .08*10**(-(25 if variant == 1 else 15)/20)/np.sqrt(np.mean(noise**2))
        stream += noise
    stream *= min(1., .95/max(float(np.max(np.abs(stream))), 1e-8))
    return np.clip(np.rint(stream*32768), -32768, 32767).astype(np.int16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'inventory', 'source-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Audit output already exists')
    study, source_root = args.study.resolve(), args.source_root.resolve()
    report = json.loads((study/'report.json').read_bytes())
    for name, key in [('plan.json', 'plan_sha256'), ('training.npz', 'pack_sha256'),
                      ('sources.json', 'sources_sha256'), ('records.json', 'records_sha256')]:
        assert sha((study/name).read_bytes()) == report[key], name
    inventory_bytes = args.inventory.read_bytes()
    assert sha(inventory_bytes) == report['inventory_sha256'] == '56dba01d2ae8f1d9bc470673012e675b1e3c7079896d491e20afef0d52108014'
    original = [json.loads(x) for x in inventory_bytes.splitlines()]
    offsets = (0, 3, 6, 9, 12)
    grouped = defaultdict(list)
    # Enumerate feasible ends rather than using the builder's source-fit flag.
    for row in original:
        signal_end = 32000+row['samples']+960
        feasible = any(all((e+d-150)*160 <= 32000 and (e+d+1)*160 >= signal_end
                           for d in offsets) for e in range(150, (signal_end+159)//160+1))
        assert feasible == row['support_geometry']['context150_offset12_rir960']['full_container']
        if feasible:
            grouped[row['sha256']].append(row)
    expected = [{'source_sha256': digest, 'aliases': sorted(aliases, key=lambda r: r['path'])}
                for digest, aliases in grouped.items()]
    expected.sort(key=lambda g: g['aliases'][0]['path'])
    assert expected == json.loads((study/'sources.json').read_bytes())
    assert len(expected) == 2594 and sum(len(g['aliases']) for g in expected) == 2632
    records = json.loads((study/'records.json').read_bytes())
    assert len(records) == 7782
    with np.load(study/'training.npz', allow_pickle=False) as bundle:
        assert set(bundle.files) == {'x', 'sequence_x', 'y'}
        windows, sequences, labels = bundle['x'], bundle['sequence_x'], bundle['y']
    assert windows.shape == (7782, 150, 40) and windows.dtype == np.uint16
    assert sequences.shape == (7782, 162, 40) and sequences.dtype == np.uint16
    assert labels.shape == (7782,) and labels.dtype == np.float32
    np.testing.assert_array_equal(windows, sequences[:, :150])
    first = {}
    for i, group in enumerate(expected):
        row = group['aliases'][0]
        first.setdefault((row['engine'], row['label']), i)
    witness_indices = set(first.values())
    from features import features
    assert sha(Path(__file__).with_name('features.py').read_bytes()) == report['frontend_source_sha256']
    witnessed, raw_members = [], set()
    for i, group in enumerate(expected):
        row = group['aliases'][0]
        assert all((a['label'], a['engine'], a['samples']) == (row['label'], row['engine'], row['samples']) for a in group['aliases'])
        relative = PurePosixPath(row['path'])
        assert relative.parts[0] == 'workspace'
        source = member(source_root, str(PurePosixPath(*relative.parts[1:])))
        assert sha(source.read_bytes()) == group['source_sha256']
        pcm = None
        if i in witness_indices:
            pcm, rate = sf.read(source, dtype='float64')
            assert rate == 16000 and pcm.shape == (row['samples'],)
        for variant in range(3):
            index = 3*i+variant
            r = records[index]
            assert (r['pack_index'], r['source_index'], r['variant']) == (index, i, variant)
            assert (r['source_group'], r['source_sha256'], r['canonical_path'], r['label']) == (row['engine'], group['source_sha256'], row['path'], row['label'])
            assert labels[index] == row['label']
            tail = 960 if variant == 2 else 0
            assert r['signal_samples'] == row['samples']+tail and r['rir_tail_samples'] == tail
            raw_members.add(r['raw'])
            raw = np.load(member(study, r['raw']), allow_pickle=False)
            assert raw.shape == ((48000+row['samples']+tail+159)//160-2, 40)
            assert raw.dtype == np.uint16 and sha(raw.tobytes()) == r['raw_sha256']
            end = r['feature_crop_end']
            assert 150 <= end and end+12 <= len(raw)
            np.testing.assert_array_equal(raw[end-150:end+12], sequences[index])
            assert sha(sequences[index].tobytes()) == r['sequence_sha256']
            assert sha(windows[index].tobytes()) == r['window_sha256']
            bounds = []
            for d in offsets:
                np.testing.assert_array_equal(raw[end+d-150:end+d], sequences[index, d:d+150])
                begin, stop = (end+d-150)*160, (end+d+1)*160
                assert begin <= 32000 and stop >= 32000+row['samples']+tail
                bounds.append([begin, stop])
            assert bounds == r['five_window_sample_bounds']
            if pcm is not None:
                rendered = reference_stream(pcm, row['path'], variant)
                assert sha(rendered.tobytes()) == r['rendered_pcm_sha256']
                np.testing.assert_array_equal(features(rendered), raw)
                witnessed.append({'pack_index': index, 'engine': row['engine'], 'label': row['label'],
                    'variant': variant, 'source': row['path'], 'rendered_pcm_sha256': r['rendered_pcm_sha256'],
                    'raw_sha256': r['raw_sha256']})
    assert {str(p.relative_to(study)) for p in (study/'raw').glob('*.npy')} == raw_members
    assert len(raw_members) == 7782 and len(witnessed) == 30
    counts = dict(Counter(f"{r['source_group']}/label{r['label']}/variant{r['variant']}" for r in records))
    assert counts == report['counts']
    result = {'role': 'independent-implementation-candidate-data-accounting',
        'source_sha256': sha(Path(__file__).read_bytes()), 'report_sha256': sha((study/'report.json').read_bytes()),
        'all_inventory_selection_and_alias_groups_match': True,
        'original_alias_rows': 2632, 'unique_canonical_sources': 2594, 'variants': 7782,
        'all_canonical_source_wav_hashes_reverified': True,
        'all_saved_raw_hashes_and_shapes_match': True, 'all_crops_exact': True,
        'five_window_crops_checked': len(records)*5,
        'all_five_conservative_sample_bounds_pass': True,
        'full_stream_reconstruction_witnesses': witnessed,
        'model_scored': False, 'training_run': False,
        'limits': ['All7782 saved raw arrays and crops audited; only30 fixed engine/label/variant witnesses independently re-rendered and frontend-extracted.',
            'Witness reconstruction uses independent Python rendering/source decoding but the same public frontend implementation, not an independent frontend.',
            'No receptive-field/streaming-score/cross-host equivalence or original pronunciation validity is proved.',
            'Candidate input data only; no training/sampling recipe, new/final scoring, playback, recording or board operation.']}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'full_stream_reconstruction_witnesses'}))


if __name__ == '__main__':
    main()
