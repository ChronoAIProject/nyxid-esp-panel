"""Save one actual CPU initialization for both arms; no forward/update calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-multispeaker-context309-initialization-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-context309-initialization-v1')
WEIGHTS = Path('.local/wakeword/mww-pretrained-reconstruction-v1/reconstructed.weights.h5')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_plan():
    import probe_hey_context_transfer_shapes_v1 as helper
    decision = R / '4090-multispeaker-coverage-context-decision-v1.json'
    audit = R / '4090-multispeaker-features-independent-audit-v1.json'
    feature = R / '4090-multispeaker-coverage-features-v1.json'
    a = json.loads(audit.read_text())
    assert a['original_report_sha256'] == sha(feature)
    assert a['all_source_role_label_geometry_checks_pass'] and a['frontend_witness_count'] == 540
    assert json.loads(decision.read_text())['context_frames'] == 309
    assert sha(WEIGHTS) == '8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'
    paths = [Path(__file__), Path(helper.__file__), Path(helper.mixednet.__file__),
             decision, audit, feature, WEIGHTS]
    plan = dict(schema='multispeaker-context309-initialization-plan-v1',
                files={str(p): sha(p) for p in paths}, tensorflow=helper.tf.__version__,
                seed=726, context_frames=309,
                policy='Load original context150 authenticated weights, transfer all non-Dense tensors exactly into context309, freeze BN. Preserve new classifier actual initialization once and bind both arms to the same saved H5/NPZ/variable inventory. No seed-based reconstruction in later runs.',
                operations='CPU architecture/assignment/save/reload only; zero forward calls, optimizer updates or feature/audio reads.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN), context_frames=309, model_calls=0)))


def run():
    import probe_hey_context_transfer_shapes_v1 as helper
    tf = helper.tf
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, digest in plan['files'].items():
        assert sha(path) == digest, path
    assert tf.__version__ == plan['tensorflow'] and not OUT.exists()
    tf.keras.utils.set_random_seed(plan['seed'])
    original = helper.create(150)
    original.load_weights(WEIGHTS)
    old = [(type(getattr(layer, 'cell', layer)).__name__, [a.copy() for a in layer.get_weights()])
           for layer in original.layers]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(plan['seed'])
    model = helper.create(plan['context_frames'])
    copied = []
    heads = []
    for i, (layer, (kind, values)) in enumerate(zip(model.layers, old, strict=True)):
        cell = getattr(layer, 'cell', layer)
        assert type(cell).__name__ == kind
        if isinstance(cell, tf.keras.layers.Dense):
            heads.append(layer)
            continue
        if values:
            layer.set_weights(values)
            assert all(np.array_equal(a, b) for a, b in zip(values, layer.get_weights(), strict=True))
            copied.append(dict(layer_index=i, kind=kind, sha256=hashlib.sha256(b''.join(a.tobytes() for a in values)).hexdigest()))
        if isinstance(cell, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            cell.trainable = False
    assert len(heads) == 1
    trainable_ids = {id(v) for v in model.trainable_variables}
    arrays = {f'v{i:03}': np.asarray(v.numpy()).copy() for i, v in enumerate(model.variables)}
    inventory = [dict(index=i, key=f'v{i:03}', name=getattr(v, 'path', v.name),
                      shape=list(v.shape), dtype=str(arrays[f'v{i:03}'].dtype),
                      effective_trainable=id(v) in trainable_ids,
                      sha256=hashlib.sha256(arrays[f'v{i:03}'].tobytes()).hexdigest())
                 for i, v in enumerate(model.variables)]
    OUT.mkdir()
    snapshot = OUT / 'initial.weights.h5'
    model.save_weights(snapshot)
    np.savez(OUT / 'variables.npz', **arrays)
    (OUT / 'variables.json').write_text(json.dumps(inventory, indent=2) + '\n')
    for v, entry in zip(model.variables, inventory, strict=True):
        assert hashlib.sha256(np.asarray(v.numpy()).tobytes()).hexdigest() == entry['sha256']
        v.assign(np.zeros(v.shape, dtype=v.numpy().dtype))
    model.load_weights(snapshot)
    with np.load(OUT / 'variables.npz', allow_pickle=False) as saved:
        for v, entry in zip(model.variables, inventory, strict=True):
            assert np.array_equal(v.numpy(), saved[entry['key']])
            assert hashlib.sha256(np.asarray(v.numpy()).tobytes()).hexdigest() == entry['sha256']
    head = heads[0].get_weights()
    result = dict(schema='multispeaker-context309-initialization-v1', plan_sha256=sha(PLAN),
                  context_frames=309, seed=plan['seed'], parameters=model.count_params(),
                  variables=len(model.variables), optimizer_variable_count_before_creation=0,
                  trainable_variables=len(model.trainable_variables), copied_encoder_bn=copied,
                  snapshot=dict(path=str(snapshot), bytes=snapshot.stat().st_size, sha256=sha(snapshot)),
                  variable_arrays=dict(path=str(OUT / 'variables.npz'), sha256=sha(OUT / 'variables.npz')),
                  variable_inventory=dict(path=str(OUT / 'variables.json'), sha256=sha(OUT / 'variables.json'), variables=inventory),
                  classifier_shapes=[list(a.shape) for a in head],
                  classifier_value_sha256=hashlib.sha256(b''.join(a.tobytes() for a in head)).hexdigest(),
                  same_instance_saved=True, actual_saved_arrays_reloaded_exact=True,
                  updates=0, inference_calls=0,
                  limits='Actual CPU initialization and saved/reload identity only. New context309 classifier differs from original150; no GPU integration, stream export or recognition qualification.')
    with (R / '4090-multispeaker-context309-initialization-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(parameters=model.count_params(), trainable_variables=len(model.trainable_variables), snapshot_sha256=sha(snapshot))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run'])
    args = parser.parse_args()
    write_plan() if args.action == 'plan' else run()
