"""Review pinned broad225 adapter with observed, constructed feature windows.

No audio, TensorFlow, private feature pack, model updates or acceptance access.
The oracle reconstructs each final tensor from observed source slices rather
than accepting shape/label counts as proof that positives survived replacement.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


PREFIX = 'research/wakeword_p4/'
TRAINER = PREFIX + 'training/train_stream.py'
ADAPTER = PREFIX + 'experiments/train_hey_broad_adapter_v1.py'
SMOKE = PREFIX + 'experiments/smoke_hey_broad_v1.py'
LOCK = PREFIX + 'results/4090-hey-broad-smoke-lock-v1.json'
TARGETS = ('hey_willow', 'hey_nyx')


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Source:
    def __init__(self, data):
        self.data = data

    def read_bytes(self):
        return self.data

    def read_text(self):
        return self.data.decode()

    def __str__(self):
        return TRAINER


class ObservedClip:
    def __init__(self, row, values, events, kind):
        self.row, self.values, self.events, self.kind = row, values, events, kind

    def __len__(self):
        return len(self.values)

    def __getitem__(self, crop):
        assert isinstance(crop, slice) and crop.step is None
        start, stop, _ = crop.indices(len(self.values))
        self.events.append((self.kind, self.row, start, stop,
                            self.values[crop].copy()))
        return self.values[crop]


class ObservedRaw:
    def __init__(self, rows, values, events):
        self.rows = {r['offset']: r for r in rows}
        self.values, self.events = values, events

    def __getitem__(self, bounds):
        row = self.rows[bounds.start]
        assert bounds.stop == row['offset'] + row['frames']
        return ObservedClip(row, self.values[row['marker']], self.events, 'source')


class PermutationRng:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.order = None

    def __getattr__(self, name):
        return getattr(self.rng, name)

    def permutation(self, size):
        self.order = self.rng.permutation(size)
        return self.order


def fixture(target):
    rows, values = [], {}
    # Unequal corpus sizes exercise equal-engine sampling independently of
    # corpus size. Whole arrays are identifiable by row and frame channels.
    for engine_index, engine in enumerate(('piper', 'kokoro', 'qwen')):
        for draw in range(engine_index + 1):
            for variant in range(3):
                for word in TARGETS:
                    rows.append(dict(engine=engine, target_id=word,
                                     pair=(engine, draw, variant)))
    for engine_index, engine in enumerate(('piper', 'kokoro', 'qwen')):
        for draw in range(engine_index + 1):
            for variant in range(3):
                rows.append(dict(engine=engine, target_id='common_negative',
                                 pair=(engine, draw, variant)))
    for i, row in enumerate(rows):
        row.update(marker=i+1, offset=i*600, frames=600, split='training',
                   lead_frame=298, keyword_frame=358+(i % 7),
                   label=int(row['target_id'] == target))
        a = np.empty((600, 40), dtype='<u2')
        a[:, 0::2] = row['marker']
        a[:, 1::2] = np.arange(600)[:, None]
        a.flags.writeable = False
        values[row['marker']] = a
    return rows, values


def check_batch(x, y, weights, events, order, target):
    ambient = [e for e in events if e[0] == 'ambient']
    source = [e for e in events if e[0] == 'source']
    assert len(ambient) == 48 and len(source) == 96
    positive, ordinary, prefix = source[:32], source[32:80], source[80:]
    assert x.shape == (128, 225, 40) and y.shape == (128, 1)
    assert weights.shape == (128,) and x.dtype == y.dtype == weights.dtype == np.float32
    for _, row, start, stop, _ in positive:
        assert row['label'] == 1 and row['target_id'] == target
        assert 3 <= stop-row['keyword_frame'] <= 27
        assert stop-start == 225 and start <= row['lead_frame']
    for _, row, start, stop, _ in ordinary:
        assert row['label'] == 0 and row['target_id'] != target
        assert max(225, row['lead_frame']-25) <= stop <= row['frames']
        assert stop-start == 225
    for _, row, start, stop, _ in prefix:
        assert row['label'] == 1 and row['target_id'] == target
        assert row['engine'] in ('piper', 'kokoro')
        assert 15 <= row['keyword_frame']-stop <= 35 and stop-start == 225
    assembled = np.stack([e[4] for e in positive+ordinary+ambient])
    assembled[32:48] = np.stack([e[4] for e in prefix])
    # Decode identifiable uint16 values; all fixture values are below clipping.
    decoded = np.rint(x * np.float32(25.6)).astype('<u2')
    assert np.array_equal(decoded, assembled[order])
    expected_labels = np.array([1]*32+[0]*96, dtype=np.float32)[:, None]
    expected_weights = np.array([1.]*32+[20.]*96, dtype=np.float32)
    assert np.array_equal(y, expected_labels[order])
    assert np.array_equal(weights, expected_weights[order])
    # Preserve all 32 own-target windows through replacement and permutation.
    assert np.array_equal(decoded[y[:, 0] == 1, :, 0].shape, (32, 225))
    schedule = []
    for stage, entries in [('ambient', ambient), ('positive', positive),
                           ('ordinary', ordinary), ('prefix', prefix)]:
        for _, row, start, stop, _ in entries:
            role = ('own' if row.get('target_id') == target else
                    'common' if row.get('target_id') == 'common_negative' else 'reciprocal')
            clock = stop-row['keyword_frame'] if stage in ('positive', 'prefix') else stop
            schedule.append((stage, row['engine'], row['pair'], role, clock))
    return schedule


def reject(call):
    try:
        call()
    except AssertionError:
        return True
    raise AssertionError('Corrupted final batch escaped the source-slice oracle')


def run_batches(functions, helpers, target, mine, augment):
    rows, values = fixture(target)
    before = digest(b''.join(v.tobytes() for v in values.values()))
    events = []
    groups = {'training': {0: {}, 1: {}}}
    for r in rows:
        groups['training'][r['label']].setdefault(r['engine'], []).append(r)
    ambient = []
    for i in range(3):
        r = dict(marker=101+i, frames=600, engine='ambient', pair=('ambient', i, 0))
        a = np.empty((600, 40), dtype='<u2')
        a[:, 0::2] = r['marker']; a[:, 1::2] = np.arange(600)[:, None]
        a.flags.writeable = False
        ambient.append((r, ObservedClip(r, a, events, 'ambient')))
    args = SimpleNamespace(source_exponent=0., negative_candidates=1,
        ambient_sampling='stratified', recover_erased_complete_coverage=False,
        prefix_negatives=16, hard_ambient_count=0, mask_prefix_loss=False,
        time_mask_width=0, frequency_mask_width=0)
    rng = PermutationRng(726)
    env = dict(helpers, np=np, args=args, frames=225, groups=groups,
        ambient={'training': ambient}, ambient_weights=np.array([.5, .3, .2]),
        raw=ObservedRaw(rows, values, events), full_source=None,
        complete_coverage=None, us_coverage=None, balanced_coverage=None,
        hard_ambient=None, rng=rng, mining_rng=np.random.default_rng(4726),
        augmentation_rng=np.random.default_rng(10726), prefix_rng=np.random.default_rng(20726),
        HEY_BATCH_GUARD=lambda *items: items)
    exec(compile(ast.Module(body=functions, type_ignores=[]), TRAINER, 'exec'), env)
    h = hashlib.sha256(); schedules = []; negatives = []
    for i in range(50):
        events.clear()
        x, y, w = env['batch'](mine=mine, augment=augment)
        schedules.append(check_batch(x, y, w, events, rng.order, target))
        for a in (x, y, w): h.update(a.tobytes())
        if i == 0:
            bad = x.copy(); pi = np.flatnonzero(y[:, 0] == 1)[0]; ni = np.flatnonzero(y[:, 0] == 0)[0]
            bad[pi] = x[ni]
            negatives.append(reject(lambda: check_batch(bad, y, w, events, rng.order, target)))
            bad_y, bad_w = y.copy(), w.copy(); bad_y[pi] = 0; bad_w[pi] = 20
            negatives.append(reject(lambda: check_batch(x, bad_y, bad_w, events, rng.order, target)))
    assert before == digest(b''.join(v.tobytes() for v in values.values()))
    return dict(target=target, mine=mine, augment=augment, batches=50,
                tensor_sha256=h.hexdigest(), input_arrays_unchanged=True,
                corrupt_positive_and_label_controls_rejected=all(negatives)), schedules


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-ref', default='2d7b566')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    def read(path):
        return subprocess.check_output(['git', 'show', commit+':'+path])
    paths = [TRAINER, ADAPTER, SMOKE, LOCK,
             PREFIX+'training/negative_sampling.py', PREFIX+'training/spec_augment.py',
             PREFIX+'results/4090-hey-broad-cpu-batch-audit-v1.json']
    sources = {path: read(path) for path in paths}
    lock = json.loads(sources[LOCK])
    for path in (TRAINER, ADAPTER, SMOKE):
        assert lock['files'][path] == digest(sources[path])
    adapter = ast.parse(sources[ADAPTER])
    pin = next(n.value.value for n in adapter.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'PIN' for t in n.targets))
    assert pin == digest(sources[TRAINER])
    cpu = json.loads(sources[paths[-1]])
    assert cpu['trainer_sha256'] == pin
    f = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == 'adapted_tree')
    env = dict(ast=ast, hashlib=hashlib, TRAINER=Source(sources[TRAINER]), PIN=pin)
    exec(compile(ast.Module(body=[f], type_ignores=[]), ADAPTER, 'exec'), env)
    tree = env['adapted_tree'](); compile(tree, TRAINER, 'exec')
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Name) and n.func.id == 'validate' for n in calls)
    guard_calls = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'HEY_BATCH_GUARD']
    assert len(guard_calls) == 1
    save = next(n for n in ast.walk(main_node) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'step == args.steps')
    assert 'model.save_weights' in ast.unparse(save)
    names = {'scale', 'sequence', 'sample', 'prefix_sample', 'batch'}
    funcs = [n for n in main_node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == names
    helpers = {}
    for path in paths[4:6]:
        exec(compile(sources[path], path, 'exec'), helpers)
    results = []; schedules = {}
    for target in TARGETS:
        for mine, augment in ((False, False), (True, True)):
            result, schedule = run_batches(funcs, helpers, target, mine, augment)
            repeat, repeat_schedule = run_batches(funcs, helpers, target, mine, augment)
            assert result == repeat and schedule == repeat_schedule
            results.append(result); schedules[(target, mine)] = schedule
        assert results[-1]['tensor_sha256'] == results[-2]['tensor_sha256']
    assert schedules[(TARGETS[0], False)] == schedules[(TARGETS[1], False)]
    result = dict(schema='mac-hey-broad-batch-flow-review-v1', gpu_commit=commit,
        source_sha256={path: digest(b) for path, b in sources.items()},
        transformed_sha256=digest(ast.unparse(tree).encode()),
        transformed_compile_pass=True, validation_calls=0, final_checkpoint_save_present=True,
        observed_batches_including_exact_repeats=400, own_target_windows_checked=12800,
        prefix_windows_checked=6400, constructed_paired_schedule_matches=True,
        results=results, limits='Constructed identifiable features exercise actual pinned adapted batch functions. No private broad feature pack, real training schedule, TensorFlow/GPU optimizer, INT8 numerical inference, recognition/FAR, audio or acceptance data accessed. Real paired schedule and trained export still require their own evidence.')
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
