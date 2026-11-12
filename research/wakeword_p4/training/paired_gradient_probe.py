"""Fifty actual inherited main/paired batches at immutable initialization."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from paired_rendering_sampler import PairedRenderingSampler
from rendering_consistency import rendering_consistency

def run_gradient_probe(model, main_batch, coverage, pack, output):
    output = Path(output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sampler = PairedRenderingSampler(pack, coverage.raw)
    head = model.layers[-1]
    assert isinstance(head, tf.keras.layers.Dense) and tuple(head.input.shape) == (None, 300)
    extractor = tf.keras.Model(model.inputs, head.input)
    variables = model.trainable_variables
    head_ids = {id(v) for v in head.trainable_variables}
    encoder = [i for i, v in enumerate(variables) if id(v) not in head_ids]
    head_indices = [i for i, v in enumerate(variables) if id(v) in head_ids]
    assert len(head_indices) == 2 and encoder
    before = [v.numpy().copy() for v in model.variables]
    bn = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert bn and all(not l.trainable for l in bn)
    expected = json.loads(Path('research/wakeword_p4/results/4090-paired-factorized-schedule-precheck-v1.json').read_text())
    traces, grad_arrays, batch_arrays = [], {}, {}

    @tf.function
    def gradients(x, y, weight, sx, sy, sw):
        with tf.GradientTape(persistent=True) as tape:
            main_e = extractor(x, training=True)
            pair_e = extractor(sx, training=True)
            main_logits = tf.matmul(main_e, head.kernel)+head.bias
            pair_logits = tf.matmul(pair_e, head.kernel)+head.bias
            main = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=main_logits)[:, 0]*weight)
            pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sy, logits=pair_logits)[:, 0]*sw)
            supervised = main+pair
            d = rendering_consistency(tf.reshape(pair_e, [2, 4, 5, 300]))
        gm = tape.gradient(main, variables); gp = tape.gradient(pair, variables)
        gs = tape.gradient(supervised, variables); gd = tape.gradient(d, variables)
        assert all(g is not None for g in gm+gp+gs)
        assert all(gd[i] is None for i in head_indices)
        assert all(gd[i] is not None for i in encoder)
        gd = [tf.zeros_like(v) if g is None else g for v, g in zip(variables, gd)]
        for g in gm+gp+gs+gd: tf.debugging.assert_all_finite(g, 'Gradient nonfinite')
        return main, pair, d, gm, gp, gs, gd, tf.linalg.norm(pair_e, axis=-1)

    accumulated = {k: [] for k in ['main', 'pair', 'supervised', 'consistency']}
    for step in range(50):
        x, y, w = main_batch(); sx, sy, sw = sampler.batch()
        trace = sampler.trace[-1]; old = expected['blocks'][step]
        assert all(trace[k] == old[k] for k in ['block', 'voices', 'speed', 'endpoint_inclusive'])
        assert trace['pack_indices'] == [v['pack_index'] for v in old['views']]
        assert trace['window_sha256'] == [v['window_sha256'] for v in old['views']]
        result = gradients(x, y, w, sx, sy, sw)
        main, pair, d = [float(v) for v in result[:3]]
        gm, gp, gs, gd = [[g.numpy() for g in values] for values in result[3:7]]
        def flat(values, indices): return np.concatenate([values[i].reshape(-1).astype(np.float64) for i in indices])
        def metrics(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return dict(supervised_norm=float(na), consistency_norm=float(nb),
                cosine=float(a@b/(na*nb)) if na and nb else None)
        item = dict(step=step+1, main_bce=main, pair_bce=pair, consistency=d,
            encoder=metrics(flat(gs, encoder), flat(gd, encoder)),
            head=metrics(flat(gs, head_indices), flat(gd, head_indices)),
            per_variable=[dict(path=v.path, **metrics(gs[i].reshape(-1).astype(np.float64), gd[i].reshape(-1).astype(np.float64))) for i, v in enumerate(variables)],
            min_embedding_norm=float(tf.reduce_min(result[7])), zero_embedding_norms=int(tf.reduce_sum(tf.cast(result[7] == 0, tf.int32))))
        for name, values in [('main', gm), ('pair', gp), ('supervised', gs), ('consistency', gd)]:
            accumulated[name].append(flat(values, range(len(variables))).astype(np.float32))
        for name, value in [('main_x', x), ('main_y', y), ('main_weight', w), ('pair_x', sx), ('pair_y', sy), ('pair_weight', sw)]:
            batch_arrays.setdefault(name, []).append(value)
        item['main_tensor_sha256'] = hashlib.sha256(x.tobytes()+y.tobytes()+w.tobytes()).hexdigest()
        item['pair_tensor_sha256'] = hashlib.sha256(sx.tobytes()+sy.tobytes()+sw.tobytes()).hexdigest()
        traces.append(item)
        if step%10 == 0: print(json.dumps(dict(probe_batches=step+1, total=50)), flush=True)
    for a, v in zip(before, model.variables): np.testing.assert_array_equal(a, v.numpy())
    assert int(model.optimizer.iterations) == 0
    # Future probes preserve this actual unchanged instance, never a seeded rebuild.
    model.save_weights(output/'probe-initialization.weights.h5')
    inventory = [dict(path=v.path, shape=list(v.shape), dtype=str(v.dtype),
        value_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in model.variables]
    (output/'probe-variable-inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    ns = np.array([r['encoder']['supervised_norm'] for r in traces])
    nd = np.array([r['encoder']['consistency_norm'] for r in traces])
    assert np.isfinite(ns).all() and np.isfinite(nd).all() and np.all(ns > 0) and np.all(nd > 0)
    ratio = float(np.median(ns)/np.median(nd))
    coefficient = min(1., 10.**np.floor(np.log10(.1*ratio)))
    np.savez_compressed(output/'probe-gradients.npz', **{k: np.stack(v) for k, v in accumulated.items()})
    np.savez_compressed(output/'probe-batches.npz', **{k: np.stack(v) for k, v in batch_arrays.items()})
    report = dict(scope='Actual original initialization, first50 unchanged main128 plus paired40 batches; no optimizer updates',
        script_sha256=sha(Path(__file__)), initialization_sha256=sha(output/'initialization.json'),
        model_snapshot_sha256=sha(output/'probe-initialization.weights.h5'),
        variable_inventory_sha256=sha(output/'probe-variable-inventory.json'),
        variable_layout=[dict(path=v.path, shape=list(v.shape), size=int(np.prod(v.shape)), encoder=i in encoder) for i, v in enumerate(variables)],
        gradients_sha256=sha(output/'probe-gradients.npz'), batches_sha256=sha(output/'probe-batches.npz'),
        main_sampler_source_sha256=sha(Path(__file__).with_name('train_stream.py')),
        paired_source_sha256=sha(Path(__file__).with_name('paired_rendering_sampler.py')),
        loss_source_sha256=sha(Path(__file__).with_name('rendering_consistency.py')),
        main_and_pair_batch_counts=50, pair_trace=sampler.trace, traces=traces,
        optimizer_updates=0, all_model_and_bn_variables_exactly_unchanged=True,
        consistency_head_disconnected=True, median_supervised_encoder_norm=float(np.median(ns)),
        median_consistency_encoder_norm=float(np.median(nd)), R=ratio, proposed_fixed_coefficient=float(coefficient),
        scaled_gradient_ratios=(coefficient*nd/ns).tolist(),
        limits='Initialization-only scale heuristic, not effectiveness/acceptance. Requires independent saved-array arithmetic audit and execution pins before training.')
    (output/'gradient-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['optimizer_updates', 'R', 'proposed_fixed_coefficient', 'median_supervised_encoder_norm', 'median_consistency_encoder_norm']}), flush=True)
