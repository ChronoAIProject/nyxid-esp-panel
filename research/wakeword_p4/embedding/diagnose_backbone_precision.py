"""Isolate backbone quantization on one existing playback/capture pair.

The recording stays private. Compare actual INT8, original float with the same
dequantized INT8 input, and original float with native mel input. All paths
retain the original output quantizer and frozen heads/threshold. No fitting,
audio transformation, live audio change or population acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(values, scale, zero):
    raw = np.rint(np.asarray(values) / scale) + zero
    clipped = int(np.count_nonzero((raw < -128) | (raw > 127)))
    return np.clip(raw, -128, 127).astype(np.int8), clipped


def float_window(blocks):
    # Source float/stream reconstruction established this phase: the first
    # fully warmed output ends at80mel frames and consumes the last76.
    if len(blocks) < 10:
        return None
    value = np.concatenate(blocks[-10:], axis=0)
    if value.shape != (80, 32):
        raise ValueError('Expected ten unchanged eight-frame mel blocks')
    return value[-76:][None, :, :, None].astype(np.float32)


def difference(a, b):
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return {'mean_absolute_byte_delta': float(np.abs(delta).mean()),
            'maximum_absolute_byte_delta': float(np.abs(delta).max()),
            'differing_bytes': int(np.count_nonzero(delta)), 'bytes': int(delta.size)}


def main():
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    import prepare_head_features as extractor
    from native_mel import NativeMel
    from evaluate_equal_mean import EqualMeanHead
    from evaluate_head import events

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if digest(Path(__file__)) != plan['script_sha256'] or args.output.exists():
        p.error('Frozen implementation and fresh private output required')
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen inference helper differs')
    base = args.data_root.resolve()
    for name, expected in plan['data_files'].items():
        path = (base / name).resolve()
        path.relative_to(base)
        if digest(path) != expected:
            raise ValueError('Frozen model/frontend/calibration differs')
    calibration = json.loads((base / plan['calibration']).read_text())
    if calibration['selected']['threshold'] != plan['threshold']:
        raise ValueError('Original validation threshold differs')
    heads = []
    for member in calibration['members']:
        path = (base / member['checkpoint'] / 'head.tflite').resolve()
        path.relative_to(base)
        if digest(path) != member['head_sha256']:
            raise ValueError('Frozen head differs')
        heads.append(path)
    model = EqualMeanHead(heads)
    frontend = NativeMel(np.load(base / plan['window']), np.load(base / plan['mel_filter']))
    extractor.initialize(base / plan['integer_model'], base / plan['window'], base / plan['mel_filter'])
    report = {'scope': __doc__, 'plan_sha256': digest(args.plan), 'cases': [],
              'threshold': plan['threshold'], 'audio_changed': False,
              'training_data_created': False, 'accepted': False,
              'limits': ['One reused source/capture pair; diagnostic, not independent recognition evidence.',
                         'Float outputs are requantized into the existing INT8-head training space. No float deployment or accuracy claim.',
                         'The first9embedding steps use the common INT8 prefix; metrics exclude them. Head scoring starts only after fully warmed float windows.',
                         'Only aggregate metrics are emitted; microphone audio, mel windows, embeddings and score sequences stay unexported.']}
    for name, path in [('source', args.source), ('capture', args.capture)]:
        spec = plan['inputs'][name]
        if digest(path) != spec['sha256']:
            raise ValueError('Existing explicit WAV differs')
        audio, rate = sf.read(path, dtype='int16')
        if rate != 16000 or audio.ndim != 1 or len(audio) != spec['samples']:
            raise ValueError('Expected unchanged mono16kHz PCM')
        primed = np.pad(audio, (plan['prime_samples'], 0))
        pcm = np.pad(primed, (480, (-len(primed)) % 1280))
        integer = Interpreter(model_path=str(base / plan['integer_model']), num_threads=1,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        reference = Interpreter(model_path=str(base / plan['float_model']), num_threads=1,
                                experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        integer.allocate_tensors()
        reference.allocate_tensors()
        di, do = integer.get_input_details()[0], integer.get_output_details()[0]
        fi, fo = reference.get_input_details()[0], reference.get_output_details()[0]
        if (tuple(di['shape']) != (1, 8, 32, 1) or tuple(fi['shape']) != (1, 76, 32, 1)
                or di['dtype'] != np.int8 or do['dtype'] != np.int8 or fi['dtype'] != np.float32):
            raise ValueError('Backbone interface differs')
        input_scale, input_zero = di['quantization']
        output_scale, output_zero = do['quantization']
        native, decoded = [], []
        paths = {k: [] for k in ('actual_int8', 'float_dequantized_input', 'float_native_input')}
        clips = {'input': 0, 'float_dequantized_output': 0, 'float_native_output': 0}
        for at in range(0, len(pcm) - 480, 1280):
            mel = frontend.block(pcm[at:at + 1760])
            quantized, count = quantize(mel, input_scale, input_zero)
            clips['input'] += count
            native.append(mel)
            decoded.append(((quantized.astype(np.float32) - input_zero) * input_scale).astype(np.float32))
            integer.set_tensor(di['index'], quantized[None, :, :, None])
            integer.invoke()
            actual = integer.get_tensor(do['index']).reshape(96)
            paths['actual_int8'].append(actual)
            for key, history, clip_key in [('float_dequantized_input', decoded, 'float_dequantized_output'),
                                           ('float_native_input', native, 'float_native_output')]:
                window = float_window(history)
                if window is None:
                    paths[key].append(actual.copy())
                    continue
                reference.set_tensor(fi['index'], window)
                reference.invoke()
                value, count = quantize(reference.get_tensor(fo['index']).reshape(96), output_scale, output_zero)
                clips[clip_key] += count
                paths[key].append(value)
        paths = {key: np.asarray(values, np.int8) for key, values in paths.items()}
        if not np.array_equal(paths['actual_int8'], extractor.extract(primed)):
            raise ValueError('Actual INT8 path does not reproduce the existing extractor exactly')
        row = {'name': name, 'wav_sha256': spec['sha256'], 'samples': len(audio),
               'steps': len(native), 'shared_prefix_steps': 9,
               'actual_int8_extraction_byte_exact': True, 'clipped_values': clips,
               'internal_quantization_difference': difference(paths['actual_int8'][9:], paths['float_dequantized_input'][9:]),
               'input_quantization_difference': difference(paths['float_dequantized_input'][9:], paths['float_native_input'][9:]),
               'paths': {}}
        original_peak = None
        for key, embeddings in paths.items():
            scores, times = model.scores(embeddings, 3)
            if original_peak is None:
                original_peak = int(scores.argmax())
            row['paths'][key] = {'peak_score': float(scores.max()),
                'peak_time_s': float(times[scores.argmax()] - plan['prime_samples'] / 16000),
                'score_at_original_peak': float(scores[original_peak]),
                'threshold_crossing_events': events(scores, plan['threshold'])}
        expected = spec.get('expected_original_peak')
        if expected is not None and not np.isclose(row['paths']['actual_int8']['peak_score'], expected, atol=1e-12, rtol=0):
            raise ValueError('Known original scalar diagnostic does not reproduce')
        report['cases'].append(row)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
