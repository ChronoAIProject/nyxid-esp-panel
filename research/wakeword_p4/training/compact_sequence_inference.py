"""Retain the compact CTC head for an offline streaming inference experiment.

This exports a research-only two-output model. It changes neither firmware nor
audio. All export/calibration/numerical-smoke inputs are original training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from phrase_ctc_readout import load_bank, sha, weights_sha


def target_log_probability(logits):
    """Sum every CTC path for [hey=1, nyx=2], blank=0; no forced choice."""
    x = np.asarray(logits, np.float64)
    if x.ndim != 3 or x.shape[-1] != 6 or not np.isfinite(x).all():
        raise ValueError('Expected finite [batch,time,6] logits')
    x = x-x.max(-1, keepdims=True)
    logp = x-np.log(np.exp(x).sum(-1, keepdims=True))
    state = np.full((len(x), 5), -np.inf)
    state[:, 0] = 0.
    for frame in logp.transpose(1, 0, 2):
        old = state
        state = np.empty_like(old)
        state[:, 0] = old[:, 0]+frame[:, 0]
        state[:, 1] = np.logaddexp(old[:, 1], old[:, 0])+frame[:, 1]
        state[:, 2] = np.logaddexp(old[:, 2], old[:, 1])+frame[:, 0]
        state[:, 3] = np.logaddexp(np.logaddexp(old[:, 3], old[:, 2]), old[:, 1])+frame[:, 2]
        state[:, 4] = np.logaddexp(old[:, 4], old[:, 3])+frame[:, 0]
    return np.logaddexp(state[:, 3], state[:, 4])


class Detector:
    def __init__(self, path):
        self.path = path
        self.reset()

    def reset(self):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        self.interpreter = Interpreter(model_path=str(self.path), num_threads=1,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        self.interpreter.allocate_tensors()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if (len(inputs) != 1 or tuple(inputs[0]['shape']) != (1, 3, 40)
                or inputs[0]['dtype'] != np.int8 or len(outputs) != 2):
            raise ValueError('Unexpected compact streaming input/output interface')
        by_shape = {tuple(item['shape']): item for item in outputs}
        if set(by_shape) != {(1, 1), (1, 67, 6)}:
            raise ValueError('Expected scalar and complete 67-state compact logits')
        self.input, self.scalar, self.sequence = inputs[0], by_shape[(1, 1)], by_shape[(1, 67, 6)]
        if any(item['dtype'] != np.uint8 or item['quantization'][0] <= 0 for item in outputs):
            raise ValueError('Expected quantized scalar and logits')

    def raw_outputs(self, raw):
        values = np.clip(np.asarray(raw, np.float32)/25.6, 0, 26)
        scale, zero = self.input['quantization']
        values = np.clip(np.rint(values/scale)+zero, -128, 127).astype(np.int8)
        scalars, sequences = [], []
        for at in range(0, len(values)-2, 3):
            self.interpreter.set_tensor(self.input['index'], values[None, at:at+3])
            self.interpreter.invoke()
            for item, result in ((self.scalar, scalars), (self.sequence, sequences)):
                q = self.interpreter.get_tensor(item['index'])[0]
                s, z = item['quantization']
                result.append((q.astype(np.float32)-z)*s)
        return np.asarray(scalars).reshape(-1), np.asarray(sequences).reshape(-1, 67, 6)

    def predict(self, raw):
        # Bound temporary output memory to 256 streaming steps. Interpreter
        # state persists across these feature chunks; no source is reset here.
        binary, combined = [], []
        for at in range(0, len(raw)-2, 768):
            p, logits = self.raw_outputs(raw[at:at+768])
            binary.extend(p)
            combined.extend(p*np.exp(target_log_probability(logits)))
        return np.asarray(binary, np.float32), np.asarray(combined, np.float64)


def build_dual(binary_weights, auxiliary_weights):
    import tensorflow as tf
    from transfer import build_model
    binary, _ = build_model(335)
    binary.load_weights(binary_weights)
    before = weights_sha(binary.weights)
    temporal = tf.keras.layers.Reshape((67, 60), name='training_phone_sequence')(
        binary.get_layer('classifier').input)
    logits = tf.keras.layers.Dense(6, name='training_phone_logits')(temporal)
    full = tf.keras.Model(binary.input, (binary.output, logits))
    full.load_weights(auxiliary_weights)
    if weights_sha(binary.weights) != before:
        raise ValueError('Auxiliary checkpoint disagrees with final binary weights')
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh export directory')
    plan = json.loads(args.plan.read_bytes())
    if any(sha(Path(__file__).parent/name) != digest for name, digest in plan['source_sha256'].items()):
        raise ValueError('Executed source differs from frozen plan')
    if any(sha(args.model/name) != digest for name, digest in plan['model_files_sha256'].items()):
        raise ValueError('Original trained model files differ')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream source differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import utils
    from microwakeword.layers import modes
    bank, labels, bank_rows = load_bank(args.bank, args.features, plan['compact_bank'])
    full = build_dual(args.model/'best.weights.h5', args.model/'phone-step-3000.weights.h5')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    streaming = utils.to_streaming_inference(full, {'stride': 3, 'spectrogram_length': 335},
                                            modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), np.float32))
    streaming.export(str(args.output/'saved_model'))
    saved = tf.saved_model.load(str(args.output/'saved_model'))
    function = saved.signatures['serving_default']
    initial = [v.numpy() for v in saved.variables]

    @tf.function(input_signature=[tf.TensorSpec([None, 40], tf.float32)])
    def scan(data):
        n = tf.shape(data)[0]//3
        p, q = tf.TensorArray(tf.float32, n), tf.TensorArray(tf.float32, n)
        def body(i, p, q):
            output = list(function(data[i*3:(i+1)*3][None]).values())
            scalar = next(v for v in output if v.shape.rank == 2)[0, 0]
            sequence = next(v for v in output if v.shape.rank == 3)[0]
            return i+1, p.write(i, scalar), q.write(i, sequence)
        _, p, q = tf.while_loop(lambda i, *_: i<n, body, (0, p, q), parallel_iterations=1)
        return p.stack(), q.stack()

    calibration_ids = np.random.default_rng(plan['calibration_seed']).choice(
        len(bank), plan['calibration_windows'], replace=False)
    digest = hashlib.sha256()
    def representative():
        for value in (0., 26.):
            item = np.full((1, 3, 40), value, np.float32)
            digest.update(item.tobytes())
            yield [item]
        for index in calibration_ids:
            value = np.clip(bank[index].astype(np.float32)/25.6, 0, 26)
            for at in range(0, len(value)-2, 3):
                item = value[None, at:at+3]
                digest.update(item.astype('<f4', copy=False).tobytes())
                yield [item]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.output/'saved_model'))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type, converter.inference_output_type = tf.int8, tf.uint8
    converter.representative_dataset = representative
    data = converter.convert()
    model_path = args.output/'compact_dual.tflite'
    model_path.write_bytes(data)
    detector = Detector(model_path)
    index = json.loads((args.features/'features.json').read_bytes())
    by_key = {(r['path'], r['variant']): r for r in index}
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    details = []
    for row_id in plan['smoke_bank_row_ids']:
        bank_row = bank_rows[row_id]
        row = by_key[bank_row['path'], bank_row['variant']]
        source = raw[row['offset']:row['offset']+row['frames']]
        scaled = np.clip(source.astype(np.float32)/25.6, 0, 26)
        for variable, value in zip(saved.variables, initial, strict=True):
            variable.assign(value)
        scalar, sequence = [np.asarray(x) for x in scan(scaled)]
        starts = 3*(np.arange(len(scalar))+1)-335
        valid = (starts >= 0) & (starts+335 <= len(source))
        if not valid.any():
            raise ValueError('No complete causal window in selected training clip')
        windows = np.stack([scaled[at:at+335] for at in starts[valid]])
        expected = [np.concatenate(v) for v in zip(*[
            [np.asarray(x) for x in full(windows[at:at+32], training=False)]
            for at in range(0, len(windows), 32)], strict=True)]
        detector.reset()
        quant_p, quant_q = detector.raw_outputs(source)
        detector.reset()
        repeat_p, repeat_q = detector.raw_outputs(source)
        if not np.array_equal(quant_p, repeat_p) or not np.array_equal(quant_q, repeat_q):
            raise ValueError('Quantized reset/replay differs')
        detector.reset()
        chunk_p, chunk_product = detector.predict(source)
        direct_product = quant_p*np.exp(target_log_probability(quant_q))
        if not np.array_equal(chunk_p, quant_p) or not np.array_equal(chunk_product, direct_product):
            raise ValueError('Chunking changes the ordered streaming outputs')
        p_delta = float(np.max(np.abs(expected[0].reshape(-1)-scalar[valid])))
        q_delta = float(np.max(np.abs(expected[1]-sequence[valid])))
        detail = {'bank_row_id': row_id, 'source_sha256': row['sha256'], 'engine': row['engine'],
            'class': int(labels[row_id]), 'compared_windows': int(valid.sum()),
            'scalar_window_stream_max_delta': p_delta, 'sequence_window_stream_max_delta': q_delta,
            'scalar_float_int8_max_delta': float(np.max(np.abs(scalar[valid]-quant_p[valid]))),
            'sequence_float_int8_max_delta': float(np.max(np.abs(sequence[valid]-quant_q[valid]))),
            'reset_replay_exact': True, 'chunking_exact': True}
        details.append(detail)
        print(json.dumps(detail), flush=True)
    passed = all(r['scalar_window_stream_max_delta'] <= plan['maximum_float_alignment_delta']
                 and r['sequence_window_stream_max_delta'] <= plan['maximum_float_alignment_delta'] for r in details)
    result = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'model_sha256': sha(model_path),
        'model_bytes': len(data), 'parameters': full.count_params(),
        'calibration_ids': calibration_ids.tolist(), 'calibration_inputs_sha256': digest.hexdigest(),
        'details': details, 'float_alignment_reset_and_chunk_smoke_passed': passed,
        'operators': sorted({op['op_name'] for op in detector.interpreter._get_ops_details()}),
        'advance_to_development_evaluation': passed, 'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'smoke_passed': passed, 'model_bytes': len(data)}), flush=True)


if __name__ == '__main__':
    main()
