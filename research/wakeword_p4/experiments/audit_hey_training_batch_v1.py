"""Run unchanged trainer batch functions on CPU, without importing TensorFlow.

This audits the base recipe only; it does not launch training, admit new material,
or certify target provenance. A paired pack additionally needs target admission.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
from negative_sampling import candidate_ends
from spec_augment import mask_spectrograms


def audit(root, target=None, return_sampler=False):
    admission = None
    if target:
        from validate_hey_target_pack_v1 import validate
        admission = validate(root, target, required_splits=('training',))
    trainer = TRAINING / 'train_stream.py'
    tree = ast.parse(trainer.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    names = {'scale', 'sequence', 'sample', 'batch'}
    funcs = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    rows = json.loads((root / 'features.json').read_text())
    raw = np.memmap(root / 'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    groups = {'training': {0: {}, 1: {}}}
    for row in rows:
        if row['split'] != 'training':
            continue
        if type(row['label']) is not int or row['label'] not in (0, 1):
            raise ValueError('Invalid label')
        offset, frames = row['offset'], row['frames']
        if type(offset) is not int or type(frames) is not int or offset < 0 or frames < 150 or offset + frames > len(raw):
            raise ValueError('Invalid feature bounds')
        if row['label'] and not 0 <= row['keyword_frame'] <= frames:
            raise ValueError('Invalid positive boundary')
        groups['training'][row['label']].setdefault(row['engine'], []).append(row)
    if not all(groups['training'].values()):
        raise ValueError('Both training labels required')
    ambient_rows = sum((json.loads((root / n).read_text()) for n in ('human.json', 'ambient.json')), [])
    ambient = {'training': []}
    for row in ambient_rows:
        if row['split'] != 'training':
            continue
        data = np.load(root / row['path'], mmap_mode='r')
        if data.ndim != 2 or data.shape[1] != 40 or len(data) < 150:
            raise ValueError('Invalid ambient shape')
        ambient['training'].append((row, data))
    if not ambient['training']:
        raise ValueError('Missing training ambient')
    args = SimpleNamespace(source_exponent=0., negative_candidates=1, ambient_sampling='rows',
                           recover_erased_complete_coverage=False, prefix_negatives=0,
                           hard_ambient_count=0, mask_prefix_loss=False,
                           time_mask_width=0, frequency_mask_width=0)
    env = dict(np=np, args=args, raw=raw, groups=groups, ambient=ambient, frames=150,
               candidate_ends=candidate_ends, mask_spectrograms=mask_spectrograms,
               full_source=None, complete_coverage=None, us_coverage=None,
               balanced_coverage=None, hard_ambient=None)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(trainer), 'exec'), env)
    if return_sampler:
        env.update(rng=np.random.default_rng(726), mining_rng=np.random.default_rng(4726),
                   augmentation_rng=np.random.default_rng(10726))
        return env
    def run():
        env.update(rng=np.random.default_rng(726), mining_rng=np.random.default_rng(4726),
                   augmentation_rng=np.random.default_rng(10726))
        return env['batch'](mine=False, augment=False)
    x, y, weights = run()
    assert x.shape == (128, 150, 40) and y.shape == (128, 1) and weights.shape == (128,)
    assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 26
    assert np.count_nonzero(y == 1) == 32 and np.count_nonzero(y == 0) == 96
    assert np.array_equal(weights, np.where(y[:, 0] == 1, 1., 20.))
    assert all(np.array_equal(a, b) for a, b in zip((x, y, weights), run()))
    return dict(schema='hey-training-base-batch-audit-v1', features=str(root),
                trainer_sha256=hashlib.sha256(trainer.read_bytes()).hexdigest(),
                extracted_functions=sorted(names), target_admission=admission,
                batch_shape=list(x.shape), positives=32, negatives=96,
                deterministic_repeat=True, batch_sha256=hashlib.sha256(x.tobytes()+y.tobytes()+weights.tobytes()).hexdigest(),
                training_launched=False, tensorflow_imported=False,
                limits='Actual unchanged base batch functions on supplied training features only; no model update, new material admission, full schedule, target adapter or acceptance certification.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--target', choices=('hey_willow', 'hey_nyx'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.features, a.target)
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
