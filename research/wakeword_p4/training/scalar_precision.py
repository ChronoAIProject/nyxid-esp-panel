"""Training-only export smoke for same-weight float and INT8 scalar streams.

No fitting, calibration change, audio device or firmware operation occurs.
Float TFLite is a diagnostic artifact, not a new P4 deployment candidate.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate import Detector as Int8Detector
from phrase_ctc_readout import sha


class Detector(Int8Detector):
    """Use the original INT8 path unchanged, or the same ordered float input."""
    def reset(self):
        super().reset()
        inputs, outputs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if (len(inputs) != 1 or len(outputs) != 1
                or tuple(self.input['shape']) != (1, 3, 40)
                or tuple(self.output['shape']) != (1, 1)):
            raise ValueError('Expected the scalar stride-three streaming interface')
        self.floating = self.input['dtype'] == np.float32 and self.output['dtype'] == np.float32
        quantized = (self.input['dtype'] == np.int8 and self.output['dtype'] == np.uint8
                     and self.scale > 0 and self.output_scale > 0)
        if not (self.floating or quantized):
            raise ValueError('Expected float32/float32 or int8/uint8 scalar inference')

    def predict(self, raw):
        if not self.floating:
            return super().predict(raw)
        values = np.clip(np.asarray(raw, np.float32)/25.6, 0, 26)
        result = []
        for at in range(0, len(values)-2, 3):
            self.interpreter.set_tensor(self.input['index'], values[None, at:at+3])
            self.interpreter.invoke()
            result.append(float(self.interpreter.get_tensor(self.output['index'])[0, 0]))
        return np.asarray(result, np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'models', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh export directory')
    plan = json.loads(args.plan.read_bytes())
    source = Path(__file__).parent
    if any(sha(source/name) != value for name, value in plan['source_sha256'].items()):
        raise ValueError('Export source differs from frozen plan')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream source differs')
    for name, value in plan['feature_index_sha256'].items():
        if sha(args.features/name) != value:
            raise ValueError('Feature index differs: '+name)
    if sha(args.features/'features.bin') != plan['feature_data_sha256']:
        raise ValueError('Full scalar feature bytes differ')
    rows = json.loads((args.features/'features.json').read_bytes())
    by_key = {(row['path'], row['variant']): row for row in rows}
    reserved = {row['sha256'] for row in rows if row['split'] != 'training'}
    if len(by_key) != len(rows):
        raise ValueError('Ambiguous feature source/render identity')
    for item in plan['smoke_sources']:
        row = by_key[item['path'], item['variant']]
        if (row['split'] != 'training' or row['sha256'] != item['sha256']
                or row['sha256'] in reserved or row['label'] != item['label']):
            raise ValueError('Smoke source is reserved or changes role')
    for arm, files in plan['model_files_sha256'].items():
        if any(sha(args.models/f'model-{arm}'/name) != value for name, value in files.items()):
            raise ValueError('Original model/SavedModel identity differs: '+arm)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    from transfer import build_model
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    reports = {}
    for arm in plan['model_files_sha256']:
        original = args.models/f'model-{arm}'
        converter = tf.lite.TFLiteConverter.from_saved_model(str(original/'saved_model'))
        converter.optimizations = []
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = converter.inference_output_type = tf.float32
        model_path = args.output/f'{arm}-float.tflite'
        model_path.write_bytes(converter.convert())
        floating = Detector(model_path)
        quantized = Detector(original/'hey_nyx.tflite')
        reference, _ = build_model(335)
        reference.load_weights(original/'best.weights.h5')
        saved = tf.saved_model.load(str(original/'saved_model'))
        signature = saved.signatures['serving_default']
        initial = [v.numpy() for v in saved.variables]

        @tf.function(input_signature=[tf.TensorSpec([None, 40], tf.float32)])
        def scan(data):
            count = tf.shape(data)[0]//3
            output = tf.TensorArray(tf.float32, count)
            def body(i, output):
                value = next(iter(signature(data[i*3:(i+1)*3][None]).values()))[0, 0]
                return i+1, output.write(i, value)
            return tf.while_loop(lambda i, _: i<count, body, (0, output), parallel_iterations=1)[1].stack()

        details = []
        for item in plan['smoke_sources']:
            row = by_key[item['path'], item['variant']]
            features = raw[row['offset']:row['offset']+row['frames']]
            scaled = np.clip(np.asarray(features, np.float32)/25.6, 0, 26)
            for variable, value in zip(saved.variables, initial, strict=True):
                variable.assign(value)
            expected = np.asarray(scan(scaled))
            floating.reset(); observed = floating.predict(features)
            floating.reset(); replay = floating.predict(features)
            floating.reset()
            chunks = np.concatenate([floating.predict(features[at:at+768])
                                     for at in range(0, len(features)-2, 768)])
            quantized.reset(); actual_int8 = quantized.predict(features)
            stock = Int8Detector(original/'hey_nyx.tflite').predict(features)
            if (not np.array_equal(observed, replay) or not np.array_equal(observed, chunks)
                    or not np.array_equal(actual_int8, stock)):
                raise ValueError('Float reset/chunking or unchanged INT8 adapter replay differs')
            starts = 3*(np.arange(len(observed))+1)-335
            valid = (starts >= 0) & (starts+335 <= len(features))
            if not valid.any():
                raise ValueError('No complete causal window in training smoke')
            windows = np.stack([scaled[start:start+335] for start in starts[valid]])
            windowed = np.concatenate([np.asarray(reference(windows[at:at+32], training=False)).reshape(-1)
                                       for at in range(0, len(windows), 32)])
            details.append({'source_sha256': row['sha256'], 'engine': row['engine'],
                'label': row['label'], 'class': item['class'], 'steps': len(observed),
                'aligned_windows': int(valid.sum()),
                'saved_float_tflite_max_delta_all_steps': float(np.abs(expected-observed).max()),
                'window_float_tflite_max_delta': float(np.abs(windowed-observed[valid]).max()),
                'float_int8_max_delta_all_steps': float(np.abs(observed-actual_int8).max()),
                'float_reset_chunking_exact': True, 'original_int8_adapter_exact': True})
            print(json.dumps({'arm': arm, **details[-1]}), flush=True)
        passed = all(max(d['saved_float_tflite_max_delta_all_steps'],
                         d['window_float_tflite_max_delta']) <= plan['maximum_float_alignment_delta']
                     for d in details)
        reports[arm] = {'float_model_sha256': sha(model_path), 'float_model_bytes': model_path.stat().st_size,
            'original_int8_model_sha256': sha(original/'hey_nyx.tflite'), 'details': details,
            'numerical_smoke_passed': passed,
            'float_operators': sorted({op['op_name'] for op in floating.interpreter._get_ops_details()})}
    result = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'arms': reports,
              'advance_to_precision_diagnostic': all(r['numerical_smoke_passed'] for r in reports.values()),
              'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'advance_to_precision_diagnostic': result['advance_to_precision_diagnostic']}), flush=True)


if __name__ == '__main__':
    main()
