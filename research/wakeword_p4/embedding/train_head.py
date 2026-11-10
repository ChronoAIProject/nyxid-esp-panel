"""Fit the prespecified frozen-backbone head using completed private features.

The explicit data root holds the v1 training/validation feature indexes; no
testing split is loaded. Generated models and PCM stay outside the repository.
"""
from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
import time
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType

HISTORY = 16
MIN_END = 25  # First nine embeddings are cold; every sampled history is warm.


def feature_directories(extra_training):
    names = ['head-features-tts-v1', 'head-features-negatives-v1']
    for entry in extra_training:
        name = entry['directory']
        if not isinstance(name, str) or Path(name).parts != (name,) or name in ('.', '..') or name in names:
            raise ValueError('Extra features must have a unique name inside the data root')
        names.append(name)
    return names


def load_rows(split, data_root, backbone_sha256, extra_training=()):
    rows = []
    names = feature_directories(extra_training)
    extras = {entry['directory']: entry for entry in extra_training}
    for name in names:
        base = data_root/name
        complete = json.loads((base/'complete.json').read_text())
        if name in extras:
            for file, key in [('index.jsonl', 'index_sha256'), ('complete.json', 'complete_sha256')]:
                if hashlib.sha256((base/file).read_bytes()).hexdigest() != extras[name][key]:
                    raise ValueError('Extra training features differ from the frozen plan')
        index = [json.loads(line) for line in (base/'index.jsonl').read_text().splitlines()]
        assert len(index) == complete['jobs']
        assert all(row['split'] in ('training', 'validation') for row in index)
        if name in extras and any(row['split'] != 'training' for row in index):
            raise ValueError('Extra training views cannot contain held-out rows')
        identity = json.loads((base/'identity.json').read_text())
        assert identity['model_sha256'] == backbone_sha256
        jobs = {job['id']: job for job in json.loads((base/'jobs-private.json').read_text())}
        for row in index:
            if row['split'] != split:
                continue
            row = dict(row)
            row['_feature_directory'] = name
            path = Path(row['path'])
            path = path if path.is_absolute() else data_root/path
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['features_sha256']:
                raise ValueError('Feature checksum mismatch')
            row['array'] = np.load(path)
            assert row['array'].shape == (row['frames'], 96)
            assert row['array'].dtype == np.int8 and row['frames'] >= MIN_END
            source = Path(jobs[row['id']]['source'])
            row['corpus'] = source.parts[source.parts.index('dataset')+1] if 'dataset' in source.parts else row['kind']
            rows.append(row)
    return rows


def make_model(architecture='conv24'):
    x = tf.keras.Input(shape=(16, 96, 1), name='embedding_history')
    if architecture in ('dnn32-layernorm', 'dnn64-layernorm', 'dnn128-layernorm'):
        width = int(architecture.split('-')[0][3:])
        y = tf.keras.layers.Flatten()(x)
        for _ in range(2):
            y = tf.keras.layers.Dense(width)(y)
            y = tf.keras.layers.LayerNormalization(epsilon=1e-5)(y)
            y = tf.keras.layers.ReLU()(y)
    elif architecture == 'conv24':
        y = tf.keras.layers.Conv2D(24, (3, 96), activation='relu')(x)
        y = tf.keras.layers.MaxPool2D((14, 1))(y)
        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(16, activation='relu')(y)
    elif architecture in ('temporal32-bn', 'temporal32-bn-dropout', 'temporal32-bn-mean-dropout',
                          'temporal32-bn-fixed-pool-dropout', 'temporal32-bn-learned-pool-dropout'):
        # Preserve order through a nine-embedding receptive field before
        # pooling positions. BatchNorm folds into the exported convolution.
        y = tf.keras.layers.Conv2D(32, (5, 96), use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization(epsilon=1e-3)(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Conv2D(32, (5, 1), use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization(epsilon=1e-3)(y)
        y = tf.keras.layers.ReLU()(y)
        if architecture == 'temporal32-bn-mean-dropout':
            y = tf.keras.layers.GlobalAveragePooling2D()(y)
        elif architecture in ('temporal32-bn-fixed-pool-dropout', 'temporal32-bn-learned-pool-dropout'):
            # Eight temporal positions remain. A uniform frozen filter is the
            # matched control; its trainable twin can learn where each feature
            # belongs in the whole phrase, instead of pooling positions away.
            y = tf.keras.layers.DepthwiseConv2D((8, 1), use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(1 / 8),
                trainable=architecture == 'temporal32-bn-learned-pool-dropout')(y)
            y = tf.keras.layers.Flatten()(y)
        else:
            y = tf.keras.layers.GlobalMaxPool2D()(y)
        if architecture.endswith('-dropout'):
            y = tf.keras.layers.Dropout(.15)(y)
    else:
        raise ValueError('Unsupported frozen head architecture')
    y = tf.keras.layers.Dense(1, activation='sigmoid')(y)
    return tf.keras.Model(x, y)


def weighted_loss(labels, predictions, recipe):
    """Optional upstream-style confidence filter; never filter evaluation."""
    per_example = tf.keras.losses.binary_crossentropy(labels, predictions)
    positive = tf.squeeze(labels, -1) > .5
    weights = tf.where(positive, 1., float(recipe['negative_weight']))
    bounds = recipe.get('loss_confidence_filter')
    if bounds is None:
        return tf.reduce_mean(per_example*weights)
    probability = tf.squeeze(predictions, -1)
    active = tf.where(positive, probability < bounds['positive_maximum'],
                      probability >= bounds['negative_minimum'])
    mask = tf.cast(active, per_example.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(per_example*weights*mask), tf.reduce_sum(mask))


class Sampler:
    def __init__(self, rows, seed, mined=None, hard_negative_stride=2):
        if type(hard_negative_stride) is not int or hard_negative_stride < 2:
            raise ValueError('Hard-negative stride must be an integer of at least two')
        self.hard_negative_stride = hard_negative_stride
        self.rng = np.random.default_rng(seed)
        self.hard_rng = np.random.default_rng(seed+1_000_000)
        self.hard = {}
        self.groups = {}
        for row in rows:
            if row['label']:
                key = ('positive', row['engine'])
            elif row['kind'] == 'tts':
                key = ('tts_negative', row['engine'])
            else:
                key = ('ambient_negative', 'all')
            self.groups.setdefault(key, []).append(row)
        self.engines = {kind: sorted(key[1] for key in self.groups if key[0] == kind)
                        for kind in ('positive', 'tts_negative', 'ambient_negative')}
        self.ambient = self.groups[('ambient_negative', 'all')]
        lengths = np.asarray([r['frames']-MIN_END+1 for r in self.ambient], np.float64)
        self.cdf = np.cumsum(lengths/lengths.sum())
        self.cdf[-1] = 1
        if mined is not None:
            by_id = {row['id']: row for row in rows}
            for record in mined:
                row = by_id[record['id']]
                if row['label'] or row['split'] != 'training' or row['features_sha256'] != record['feature_sha256']:
                    raise ValueError('Mined window is not a verified training negative')
                end = record['end']
                if not isinstance(end, int) or not MIN_END <= end <= row['frames']:
                    raise ValueError('Mined window is outside warmed history')
                key = ('tts_negative', row['engine']) if row['kind'] == 'tts' else ('ambient_negative', 'all')
                self.hard.setdefault(key, []).append((row, end))
            if not any(key[0] == 'tts_negative' for key in self.hard) or ('ambient_negative', 'all') not in self.hard:
                raise ValueError('Both mined negative categories must be populated')

    def batch(self):
        examples, labels = [], []
        for kind, count in [('positive', 32), ('tts_negative', 48), ('ambient_negative', 48)]:
            for _ in range(count):
                if kind == 'ambient_negative':
                    row = self.ambient[int(np.searchsorted(self.cdf, self.rng.random()))]
                else:
                    engine = self.rng.choice(self.engines[kind])
                    group = self.groups[(kind, engine)]
                    row = group[int(self.rng.integers(len(group)))]
                if kind == 'positive':
                    end = int(np.ceil((row['keyword_end']+self.rng.uniform(.03, .32))/.08))
                    if end < MIN_END or end > row['frames']:
                        raise ValueError('Positive endpoint lies outside a warmed history')
                else:
                    end = int(self.rng.integers(MIN_END, row['frames']+1))
                if self.hard and kind != 'positive' and _ % self.hard_negative_stride == 0:
                    # Consume the original random draws first. The separate
                    # hard RNG keeps positives/unreplaced negatives identical to control.
                    engines = sorted(key[1] for key in self.hard if key[0] == kind)
                    engine = self.hard_rng.choice(engines)
                    group = self.hard[(kind, engine)]
                    row, end = group[int(self.hard_rng.integers(len(group)))]
                examples.append(row['array'][end-HISTORY:end])
                labels.append(kind == 'positive')
        x = np.asarray(examples, np.float32)[:, :, :, None]/128
        y = np.asarray(labels, np.float32)[:, None]
        order = self.rng.permutation(len(y))
        return x[order], y[order]


def export(model, output, examples):
    model.save(output/'head.keras')
    model.export(str(output/'saved-model'), input_signature=[
        tf.TensorSpec((1, 16, 96, 1), tf.float32, name='embedding_history')])
    converter = tf.lite.TFLiteConverter.from_saved_model(str(output/'saved-model'))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative():
        for x in examples:
            yield [x[None]]
        # This is the physical byte range, not fitted validation/test extrema.
        yield [np.full((1, 16, 96, 1), -1, np.float32)]
        yield [np.full((1, 16, 96, 1), 127/128, np.float32)]

    converter.representative_dataset = representative
    data = converter.convert()
    (output/'head.tflite').write_bytes(data)
    interpreter = Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    di, do = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert tuple(di['shape']) == (1, 16, 96, 1) and tuple(do['shape']) == (1, 1)
    assert di['quantization'] == (1/128, 0), di['quantization']
    integers, raw_outputs = [], []
    for x in examples[:64]:
        interpreter.set_tensor(di['index'], np.rint(x[None]*128).astype(np.int8))
        interpreter.invoke()
        raw = interpreter.get_tensor(do['index']).copy()
        raw_outputs.append(raw[0])
        integers.append(float((int(raw[0, 0])-do['quantization'][1])*do['quantization'][0]))
    floats = np.asarray(model(examples[:64], training=False)).reshape(-1)
    np.save(output/'numeric-inputs.npy', np.rint(examples[:64]*128).astype(np.int8))
    np.save(output/'numeric-expected.npy', np.asarray(raw_outputs, np.int8))
    report = {'model_sha256': hashlib.sha256(data).hexdigest(), 'model_bytes': len(data),
              'input_quantization': di['quantization'], 'output_quantization': do['quantization'],
              'float_int8_max_absolute_error': float(np.max(np.abs(floats-integers))),
              'float_int8_mean_absolute_error': float(np.mean(np.abs(floats-integers))),
              'ops': [op['op_name'] for op in interpreter._get_ops_details()]}
    (output/'export.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--initial-weights', type=Path)
    parser.add_argument('--mined-negatives', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text())
    recipe = plan['recipe']
    bounds = recipe.get('loss_confidence_filter')
    if bounds is not None and not 0 < bounds['negative_minimum'] < bounds['positive_maximum'] < 1:
        raise ValueError('Invalid training-only confidence filter')
    for path, field in [(args.initial_weights, 'initial_weights_sha256'),
                        (args.mined_negatives, 'mined_negatives_sha256')]:
        if bool(path) != bool(plan.get(field)):
            raise ValueError('Initialization/mining arguments differ from the frozen plan')
        if path and hashlib.sha256(path.read_bytes()).hexdigest() != plan[field]:
            raise ValueError('Initialization/mining checksum differs from the frozen plan')
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(plan['recipe']['seed'])
    started = time.monotonic()
    extra_training = plan.get('extra_training_features', [])
    rows = load_rows('training', args.data_root, plan['model_sha256'], extra_training)
    # Check source-level split disjointness without loading validation embeddings.
    validation_hashes = set()
    for name in feature_directories(extra_training):
        for line in (args.data_root/name/'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] == 'validation':
                validation_hashes.add(row['source_sha256'])
    assert not validation_hashes.intersection(r['source_sha256'] for r in rows)
    mined = json.loads(args.mined_negatives.read_text()) if args.mined_negatives else None
    if mined:
        assert mined['backbone_sha256'] == plan['model_sha256']
        for name, digest in mined['feature_indexes'].items():
            assert hashlib.sha256((args.data_root/name/'index.jsonl').read_bytes()).hexdigest() == digest
    sampler = Sampler(rows, plan['recipe']['seed'], mined['records'] if mined else None,
                      recipe.get('hard_negative_stride', 2))
    representative_rows = [row for row in rows if row['_feature_directory'] in
                           ('head-features-tts-v1', 'head-features-negatives-v1')]
    representative_sampler = Sampler(representative_rows, plan['recipe']['seed']+1)
    examples = np.concatenate([representative_sampler.batch()[0] for _ in range(4)])
    np.save(args.output/'representatives-training-only.npy', examples)
    model = make_model(plan.get('architecture', 'conv24'))
    if args.initial_weights:
        initial = tf.keras.models.load_model(args.initial_weights)
        model.set_weights(initial.get_weights())
        assert all(np.array_equal(a, b) for a, b in zip(model.get_weights(), initial.get_weights()))
        del initial
    optimizer = tf.keras.optimizers.Adam(plan['recipe']['learning_rate'])

    @tf.function(input_signature=[tf.TensorSpec((128, 16, 96, 1), tf.float32), tf.TensorSpec((128, 1), tf.float32)])
    def step(x, y):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = weighted_loss(y, prediction, recipe)
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, tf.reduce_mean(tf.cast((prediction >= .5) == (y >= .5), tf.float32))

    summary = {'scope': 'Frozen-backbone training only; no recognition acceptance', 'parameters': model.count_params(),
               'plan': plan, 'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'training_sources': len(rows),
               'training_frames': sum(r['frames'] for r in rows), 'checkpoints': []}
    summary['initial_weights_sha256'] = plan.get('initial_weights_sha256')
    summary['mined_negatives_sha256'] = plan.get('mined_negatives_sha256')
    summary['mined_windows'] = len(mined['records']) if mined else 0
    summary['feature_inputs'] = {}
    for name in feature_directories(extra_training):
        base = args.data_root/name
        summary['feature_inputs'][name] = {
            'index_sha256': hashlib.sha256((base/'index.jsonl').read_bytes()).hexdigest(),
            'identity': json.loads((base/'identity.json').read_text()),
            'completion': json.loads((base/'complete.json').read_text())}
    (args.output/'training.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'phase': 'loaded', 'rows': len(rows), 'parameters': model.count_params(), 'seconds': time.monotonic()-started}), flush=True)
    for number in range(1, plan['recipe']['steps']+1):
        fraction = number/recipe['steps']
        factor = recipe['learning_rate_eighty_percent_factor'] if fraction > .8 else (
            recipe['learning_rate_half_factor'] if fraction > .5 else 1)
        lr = recipe['learning_rate']*factor
        optimizer.learning_rate.assign(lr)
        x, y = sampler.batch()
        loss, accuracy = step(x, y)
        if number % 100 == 0:
            print(json.dumps({'step': number, 'loss': float(loss), 'batch_accuracy': float(accuracy),
                              'learning_rate': lr, 'seconds': time.monotonic()-started}), flush=True)
        if number in plan['recipe']['checkpoints']:
            output = args.output/f'step-{number:05d}'
            output.mkdir()
            report = export(model, output, examples)
            summary['checkpoints'].append({'step': number, **report})
            (args.output/'training.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(json.dumps({'phase': 'exported', 'step': number, **report}), flush=True)
    summary['seconds'] = time.monotonic()-started
    (args.output/'complete.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
