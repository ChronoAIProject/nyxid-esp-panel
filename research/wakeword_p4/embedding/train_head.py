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


def load_rows(split, data_root, backbone_sha256):
    rows = []
    for name in ('head-features-tts-v1', 'head-features-negatives-v1'):
        base = data_root/name
        complete = json.loads((base/'complete.json').read_text())
        index = [json.loads(line) for line in (base/'index.jsonl').read_text().splitlines()]
        assert len(index) == complete['jobs']
        assert all(row['split'] in ('training', 'validation') for row in index)
        identity = json.loads((base/'identity.json').read_text())
        assert identity['model_sha256'] == backbone_sha256
        jobs = {job['id']: job for job in json.loads((base/'jobs-private.json').read_text())}
        for row in index:
            if row['split'] != split:
                continue
            row = dict(row)
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


def make_model():
    x = tf.keras.Input(shape=(16, 96, 1), name='embedding_history')
    y = tf.keras.layers.Conv2D(24, (3, 96), activation='relu')(x)
    y = tf.keras.layers.MaxPool2D((14, 1))(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(16, activation='relu')(y)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(y)
    return tf.keras.Model(x, y)


class Sampler:
    def __init__(self, rows, seed):
        self.rng = np.random.default_rng(seed)
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
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text())
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(plan['recipe']['seed'])
    started = time.monotonic()
    rows = load_rows('training', args.data_root, plan['model_sha256'])
    # Check source-level split disjointness without loading validation embeddings.
    validation_hashes = set()
    for name in ('head-features-tts-v1', 'head-features-negatives-v1'):
        for line in (args.data_root/name/'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] == 'validation':
                validation_hashes.add(row['source_sha256'])
    assert not validation_hashes.intersection(r['source_sha256'] for r in rows)
    sampler = Sampler(rows, plan['recipe']['seed'])
    representative_sampler = Sampler(rows, plan['recipe']['seed']+1)
    examples = np.concatenate([representative_sampler.batch()[0] for _ in range(4)])
    np.save(args.output/'representatives-training-only.npy', examples)
    model = make_model()
    optimizer = tf.keras.optimizers.Adam(plan['recipe']['learning_rate'])

    @tf.function(input_signature=[tf.TensorSpec((128, 16, 96, 1), tf.float32), tf.TensorSpec((128, 1), tf.float32)])
    def step(x, y):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            loss = tf.keras.losses.binary_crossentropy(y, prediction)
            weights = tf.where(tf.squeeze(y, -1) > .5, 1., 5.)
            loss = tf.reduce_mean(loss*weights)
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, tf.reduce_mean(tf.cast((prediction >= .5) == (y >= .5), tf.float32))

    summary = {'scope': 'Frozen-backbone training only; no recognition acceptance', 'parameters': model.count_params(),
               'plan': plan, 'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'training_sources': len(rows),
               'training_frames': sum(r['frames'] for r in rows), 'checkpoints': []}
    summary['feature_inputs'] = {}
    for name in ('head-features-tts-v1', 'head-features-negatives-v1'):
        base = args.data_root/name
        summary['feature_inputs'][name] = {
            'index_sha256': hashlib.sha256((base/'index.jsonl').read_bytes()).hexdigest(),
            'identity': json.loads((base/'identity.json').read_text()),
            'completion': json.loads((base/'complete.json').read_text())}
    (args.output/'training.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'phase': 'loaded', 'rows': len(rows), 'parameters': model.count_params(), 'seconds': time.monotonic()-started}), flush=True)
    for number in range(1, plan['recipe']['steps']+1):
        lr = .001*(.1 if number > 8000 else .3 if number > 5000 else 1)
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
