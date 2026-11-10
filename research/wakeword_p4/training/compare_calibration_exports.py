"""Controlled export-only comparison: real training features vs forced extrema.

Weights, SavedModel initial state and actual representative sequences are fixed.
Only the legacy zero/full26 calibration prefix differs. No classifier selection
or validation/final audio is used here.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-group', type=int, default=12)
    a = p.parse_args()
    if a.output.exists() or a.per_group < 1:
        p.error('Fresh output and positive per-group required')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    base = json.loads((a.model_dir/'training.json').read_text())
    frames = base['context_frames']
    if frames % 3:
        raise ValueError('Calibration needs complete three-frame slices')
    indices = {name: json.loads((a.features/name).read_text())
               for name in ('features.json', 'human.json', 'ambient.json')}
    if sha(a.features/'features.json') != base['features_index_sha256']:
        raise ValueError('Different training keyword index')
    groups = defaultdict(list)
    for row in indices['features.json']:
        if row['split'] == 'training' and row['frames'] >= frames:
            groups[f"keyword:{row['engine']}:{row['label']}"].append(('keyword', row))
    for name in ('human.json', 'ambient.json'):
        for row in indices[name]:
            if row['split'] == 'training':
                count = len(np.load(a.features/row['path'], mmap_mode='r', allow_pickle=False))
                if count >= frames:
                    groups[f"{name}:{row.get('category', 'unspecified')}"].append(('ambient', dict(row, frames=count)))
    rng = np.random.default_rng(728)
    raw = np.memmap(a.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    sequences, provenance = [], []
    for key, rows in sorted(groups.items()):
        for _ in range(a.per_group):
            kind, row = rows[int(rng.integers(len(rows)))]
            offset = int(rng.integers(row['frames']-frames+1))
            if kind == 'keyword':
                segment = raw[row['offset']+offset:row['offset']+offset+frames]
                digest = None  # Complete shared binary SHA is recorded below.
            else:
                path = (a.features/row['path']).resolve()
                # Existing composite datasets intentionally symlink audited caches.
                digest = sha(path)
                if row.get('feature_sha256') and digest != row['feature_sha256']:
                    raise ValueError('Changed ambient feature array')
                segment = np.load(path, mmap_mode='r', allow_pickle=False)[offset:offset+frames]
            values = np.clip(segment.astype(np.float32)/25.6, 0, 26)
            if values.shape != (frames, 40) or not np.isfinite(values).all():
                raise ValueError('Invalid waveform-derived representative features')
            sequences.append(values)
            provenance.append(dict(group=key, path=row['path'], split=row['split'], variant=row.get('variant'),
                                   relative_frame_offset=offset, array_sha256=digest,
                                   feature_slice_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    if not sequences:
        raise ValueError('No training sequences')
    a.output.mkdir(parents=True)
    data = np.stack(sequences)
    np.save(a.output/'representative.npy', data)
    saved_files = {str(p.relative_to(a.model_dir/'saved_model')): sha(p)
                   for p in sorted((a.model_dir/'saved_model').rglob('*')) if p.is_file()}
    report = dict(role='Controlled training-only INT8 export calibration ablation; no acceptance',
                  source_sha256=sha(Path(__file__)), parent_training_report_sha256=sha(a.model_dir/'training.json'),
                  parent_weights_sha256=sha(a.model_dir/'best.weights.h5'), saved_model_files_sha256=saved_files,
                  feature_indices_sha256={name: sha(a.features/name) for name in indices},
                  feature_binary_sha256=sha(a.features/'features.bin'), seed=728, context_frames=frames,
                  sequences=len(data), group_counts={k: a.per_group for k in sorted(groups)},
                  representative_sha256=sha(a.output/'representative.npy'), records=provenance,
                  minimum=float(data.min()), maximum=float(data.max()), exports=[],
                  original_model_unchanged=True, weights_retrained=False,
                  validation_audio_accessed=False, independent_final_audio_accessed=False,
                  limits='Real training-derived features only. Same SavedModel and saved initial state; control differs solely by two legacy artificial extrema inputs. New representative sampling differs from original training export; causal comparison is between these two new exports, not either vs original. Calibration does not prove quality; subsequent validation must retain cold-start events and all original threshold gates.')
    for policy in ('with-extrema', 'representative'):
        out = a.output/policy
        out.mkdir()

        def representative():
            if policy == 'with-extrema':
                yield [np.zeros((1, 3, 40), dtype=np.float32)]
                yield [np.full((1, 3, 40), 26, dtype=np.float32)]
            for sequence in data:
                for at in range(0, frames, 3):
                    yield [sequence[None, at:at+3]]

        converter = tf.lite.TFLiteConverter.from_saved_model(str(a.model_dir/'saved_model'))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter._experimental_variable_quantization = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative
        (out/'hey_nyx.tflite').write_bytes(converter.convert())
        for name in ('saved_model', 'best.weights.h5'):
            (out/name).symlink_to((a.model_dir/name).resolve(), target_is_directory=name == 'saved_model')
        exported = dict(policy=policy, model_sha256=sha(out/'hey_nyx.tflite'), model_bytes=(out/'hey_nyx.tflite').stat().st_size)
        (out/'training.json').write_text(json.dumps(dict(base, **exported,
            parent_training_report_sha256=report['parent_training_report_sha256'], weights_retrained=False,
            export_only_ablation=True), indent=2)+'\n')
        report['exports'].append(exported)
        print(json.dumps(exported), flush=True)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
