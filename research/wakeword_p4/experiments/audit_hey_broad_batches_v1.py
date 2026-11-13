"""CPU admission of original batch functions at225; no training or scoring."""
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
from ambient_sampling import stratified_probabilities


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def audit(target):
    root = Path('.local/wakeword/hey-broad-training-features-v1') / target
    provenance = json.loads((root/'target-provenance.json').read_text())
    assert provenance['schema'] == 'hey-broad-context225-target-pack-v1'
    assert provenance['target_id'] == target and provenance['context_frames'] == 225
    for name, expected in provenance['files'].items(): assert sha(root/name) == expected
    rows = json.loads((root/'features.json').read_text())
    groups = {'training': {0: {}, 1: {}}}
    positive_ends = prefix_ends = 0
    for row in rows:
        assert row['split'] == 'training' and row['label'] == int(row['target_id'] == target)
        assert row['frames'] >= 225
        groups['training'][row['label']].setdefault(row['engine'], []).append(row)
        if row['label']:
            for jitter in range(3, 28):
                end = row['keyword_frame'] + jitter
                assert 225 <= end <= row['frames'] and end-225 <= row['lead_frame']
                positive_ends += 1
            if row['engine'] in ('piper', 'kokoro'):
                for delta in range(15, 36):
                    end = row['keyword_frame'] - delta
                    assert 225 <= end <= row['frames']
                    prefix_ends += 1
    ambient_rows = sum((json.loads((root/name).read_text()) for name in ('human.json', 'ambient.json')), [])
    assert all(r['split'] == 'training' for r in ambient_rows)
    ambient = {'training': [(r, np.load(root/r['path'], mmap_mode='r')) for r in ambient_rows]}
    assert all(a.ndim == 2 and a.shape[1] == 40 and len(a) >= 225 for _, a in ambient['training'])
    trainer = TRAINING/'train_stream.py'
    main = next(n for n in ast.parse(trainer.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    names = {'scale', 'sequence', 'sample', 'prefix_sample', 'batch'}
    funcs = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    prefix = next(n for n in funcs if n.name == 'prefix_sample')
    original = ast.parse('sources = groups["training"][1]').body[0]
    assert ast.dump(prefix.body[0]) == ast.dump(original)
    prefix.body[0] = ast.parse('sources = {e: r for e, r in groups["training"][1].items() if e in ("piper", "kokoro")}').body[0]
    args = SimpleNamespace(source_exponent=0., negative_candidates=1, ambient_sampling='stratified',
        recover_erased_complete_coverage=False, prefix_negatives=16, hard_ambient_count=0,
        mask_prefix_loss=False, time_mask_width=0, frequency_mask_width=0)
    env = dict(np=np, args=args, frames=225, groups=groups, ambient=ambient,
        ambient_weights=stratified_probabilities(ambient_rows),
        raw=np.memmap(root/'features.bin', mode='r', dtype='<u2').reshape(-1, 40),
        candidate_ends=candidate_ends, mask_spectrograms=mask_spectrograms,
        full_source=None, complete_coverage=None, us_coverage=None, balanced_coverage=None, hard_ambient=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(trainer), 'exec'), env)
    def run():
        for name, seed in [('rng',726), ('mining_rng',4726), ('augmentation_rng',10726), ('prefix_rng',20726)]:
            env[name] = np.random.default_rng(seed)
        h = hashlib.sha256()
        for _ in range(50):
            x, y, w = env['batch'](mine=False, augment=False)
            assert x.shape == (128,225,40) and y.shape == (128,1) and w.shape == (128,)
            assert np.isfinite(x).all() and 0 <= x.min() <= x.max() <= 26
            assert np.count_nonzero(y) == 32 and np.array_equal(w, np.where(y[:,0] == 1, 1., 20.))
            for a in (x,y,w): h.update(a.tobytes())
        return h.hexdigest()
    first = run(); assert run() == first
    return dict(target=target, positive_ends_enumerated=positive_ends, prefix_ends_enumerated=prefix_ends,
        batches=50, repeat_exact=True, tensor_sha256=first, provenance_sha256=sha(root/'target-provenance.json'))


if __name__ == '__main__':
    result = dict(schema='hey-broad-context225-cpu-batch-audit-v1', script_sha256=sha(__file__),
        trainer_sha256=sha(TRAINING/'train_stream.py'), results=[audit(t) for t in ('hey_willow','hey_nyx')],
        prefix_policy='16 slots, only Piper/Kokoro; equal eligible-engine probability. Generated duration boundaries are synthetic proxies, not phonetic truth. Qwen remains in ordinary positive and negative draws.',
        limits='CPU batch admission only; no model, optimizer, full schedule, paired-draw certification or INT8 export. No augmentation; training must explicitly bind these functions and policy.')
    path = Path('research/wakeword_p4/results/4090-hey-broad-cpu-batch-audit-v1.json')
    with path.open('x') as f: f.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
