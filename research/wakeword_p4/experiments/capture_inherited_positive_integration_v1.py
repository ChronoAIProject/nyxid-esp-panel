"""Observe inherited-positive recovery with original draw identities and two updates.

Run in a separate process and evaluation directory with the original training
packs. This observes the real batch function and optimizer; it supplies no
replacement features, sampler, loss, model or initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


BASE_SHA = '2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904'
RECOVERY_SHA = 'b97b2debc92ca106ab199af60ac144aa7a78ce4a26e39621bc5b179fb637917e'


class ProbeComplete(Exception):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x))


def rng_states(local):
    result = {}
    for name, value in local.items():
        if isinstance(value, np.random.Generator):
            result[name] = value.bit_generator.state
    for name in ('full_source', 'complete_coverage', 'us_coverage', 'balanced_coverage', 'paired_rendering'):
        obj = local.get(name)
        if obj is None:
            continue
        for prefix, item in ((name, obj), (name + '.sampler', getattr(obj, 'sampler', None))):
            if item is not None:
                for key, value in vars(item).items():
                    if isinstance(value, np.random.Generator):
                        result[prefix + '.' + key] = value.bit_generator.state
    # Detach state dictionaries from all live objects.
    return json.loads(encoded(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trainer', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    trainer = args.trainer.resolve()
    identity = sha(trainer)
    assert identity in (BASE_SHA, RECOVERY_SHA), 'Unexpected trainer bytes'
    assert not args.output.exists(), 'Preserve prior/failed outputs'
    config = json.loads(args.config.read_text())
    cli = config['trainer_args']
    assert isinstance(cli, list) and all(isinstance(item, str) for item in cli)
    flags = [item for item in cli if item.startswith('--')]
    assert len(flags) == len(set(flags)), 'Duplicate trainer options'
    assert '--output' not in cli and '--paired-gradient-probe' not in cli
    assert cli[cli.index('--steps') + 1] == '120000'
    assert cli[cli.index('--eval-interval') + 1] == '5000'
    assert cli[cli.index('--paired-objective') + 1] == 'phrase-ranking'
    assert cli[cli.index('--paired-rendering-coefficient') + 1] == '1'
    assert cli[cli.index('--paired-draw-noise-condition') + 1] in ('fixed', 'per_draw')
    if '--recover-inherited-positive-draws' in cli:
        assert identity == RECOVERY_SHA
    # Original steps/eval schedule prevents a two-step learning-rate schedule
    # from replacing the formal run's first two updates. The observer stops
    # after update2; validate() and export must never be entered.
    source = trainer.read_text().splitlines()
    before_order = next(i + 1 for i, line in enumerate(source) if line == '        order = rng.permutation(len(x))')
    after_update = next(i + 1 for i, line in enumerate(source) if line == '            if step % 500 == 0:')
    before_full_source = next(i + 1 for i, line in enumerate(source) if line == '        if full_source is not None:')
    positive_append = next(i + 1 for i, line in enumerate(source) if line == '                output.append(data[end-frames:end])')
    main_frame = None
    batches, updates, positive_sources = [], [], []
    pending = None
    initial_positive = None
    current_sources = []

    def observe(frame, event, value):
        nonlocal main_frame, pending, initial_positive, current_sources
        if frame.f_code.co_filename != str(trainer):
            return None
        name = frame.f_code.co_name
        if name in ('validate', 'representative') and event == 'call':
            raise RuntimeError('Integration probe must not validate or export')
        if name == 'main':
            main_frame = frame
            if event == 'line' and frame.f_lineno == after_update:
                local = frame.f_locals
                step = int(local['step'])
                assert int(local['model'].optimizer.iterations) == step
                values = [float(v) for v in local['values']]
                assert len(values) == 6 and np.isfinite(values).all()
                updates.append(dict(step=step, values=values, rng_states=rng_states(local)))
                if step == 2:
                    raise ProbeComplete()
        elif name == 'sample':
            if event == 'line' and frame.f_lineno == positive_append and frame.f_locals['label']:
                local = frame.f_locals
                row = json.loads(encoded(local['row']))
                assert row['split'] == 'training' and row['label'] == 1
                row.update(actual_crop_start_frame=int(local['end']) - 150, actual_crop_stop_frame=int(local['end']))
                current_sources.append(row)
        elif name == 'batch':
            if event == 'line' and frame.f_lineno == before_full_source:
                initial_positive = frame.f_locals['x'][:32].copy()
                assert len(current_sources) == 32
            if event == 'line' and frame.f_lineno == before_order:
                pending = {key: frame.f_locals[key].copy() for key in ('x', 'y', 'weights')}
                if 'recovered_complete' in frame.f_locals:
                    pending['recovered_x'] = frame.f_locals['recovered_complete'].copy()
            elif event == 'return':
                assert pending is not None and len(batches) < 2
                x, y, weights = value
                order = frame.f_locals['order']
                np.testing.assert_array_equal(x, pending['x'][order])
                np.testing.assert_array_equal(y, pending['y'][order])
                np.testing.assert_array_equal(weights, pending['weights'][order])
                observed = dict(pre_x=pending['x'], pre_y=pending['y'], pre_weight=pending['weights'],
                                order=order.copy(), main_x=x.copy(), main_y=y.copy(), main_weight=weights.copy(),
                                initial_positive_x=initial_positive.copy())
                if 'recovered_x' in pending:
                    observed['recovered_x'] = pending['recovered_x']
                batches.append(observed)
                positive_sources.append(current_sources)
                current_sources = []
                initial_positive = None
                pending = None
        return observe

    sys.path.insert(0, str(trainer.parent))
    prior_argv = sys.argv
    sys.argv = [str(trainer), *cli, '--output', str(args.output.resolve())]
    try:
        sys.settrace(observe)
        try:
            runpy.run_path(str(trainer), run_name='__main__')
        except ProbeComplete:
            assert len(batches) == len(updates) == 2
        else:
            raise RuntimeError('Trainer returned before the two-update boundary')
    finally:
        sys.settrace(None)
        sys.argv = prior_argv
        if main_frame is not None:
            for name in ('paired_trace_log', 'paired_loss_log'):
                stream = main_frame.f_locals.get(name)
                if stream is not None:
                    stream.flush()
                    stream.close()
    local = main_frame.f_locals
    from batchnorm_policy import snapshot
    assert snapshot(local['model']) == local['bn_initial']
    variables = [v.numpy() for v in local['model'].variables]
    assert all(np.isfinite(v).all() for v in variables)
    assert not (args.output / 'training.json').exists()
    assert not (args.output / 'best.weights.h5').exists()
    assert not list(args.output.glob('*.tflite'))
    for index, batch in enumerate(batches, 1):
        np.savez_compressed(args.output / f'observed-network-input-step{index}.npz', **batch)
    report = dict(schema='inherited-positive-real-trainer-two-update-probe-v1',
                  capture_script_sha256=sha(Path(__file__)), trainer_sha256=identity,
                  config_sha256=sha(args.config), optimizer_updates=2,
                  source_hashes=local['source_hashes'], feature_metadata_hashes=local['feature_metadata_hashes'],
                  initialization_inventory_sha256=sha(args.output / 'actual-initialization-inventory.json'),
                  final_model_values_sha256=hashlib.sha256(b''.join(v.tobytes() for v in variables)).hexdigest(),
                  frozen_batchnorm_unchanged=True, all_model_variables_finite=True,
                  steps=updates, original_positive_sources_by_step=positive_sources,
                  inherited_positive_recovery_enabled='--recover-inherited-positive-draws' in cli,
                  complete_recovery_enabled='--recover-erased-complete-coverage' in cli,
                  validation_invoked=False, export_invoked=False,
                  acceptance_claim=False,
                  scope='Actual main128/paired40 optimizer integration only; not a trained candidate or quality result.')
    (args.output / 'integration-probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(optimizer_updates=2, frozen_batchnorm_unchanged=True, acceptance_claim=False)))


if __name__ == '__main__':
    main()
