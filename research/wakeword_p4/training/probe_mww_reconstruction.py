"""Check teacher-target responses and intermediate activations, never Nyx accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from features import features
from reconstruct_mww_pretrained import ARCH, SOURCE_SHA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'model', 'reconstruction', 'audio', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh directory')
    if sha(a.model) != SOURCE_SHA:
        raise ValueError('Unexpected teacher model')
    generation = json.loads((a.audio/'generation.json').read_text())
    if sha(a.audio/'manifest.json') != generation['manifest_sha256']:
        raise ValueError('Changed audio manifest')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    sys.path.insert(0, str(a.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
    model = mixednet.model(parser.parse_args(ARCH), (150, 40), None)
    model.load_weights(a.reconstruction/'reconstructed.weights.h5')
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    streaming = utils.to_streaming_inference(model, {'stride': 3, 'spectrogram_length': 150},
                                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    tap_layers = [l for l in streaming.layers if isinstance(l, (tf.keras.layers.Activation, tf.keras.layers.DepthwiseConv2D))]
    probe = tf.keras.Model(streaming.inputs, [l.output for l in tap_layers]+[streaming.output])
    probe(np.zeros((1, 3, 40), np.float32), training=False)
    states = [v for v in probe.variables if 'states' in v.name]
    if len(states) != 6 or len(tap_layers) != 9:
        raise ValueError('Unexpected graph/state structure')
    @tf.function(input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
    def infer(x):
        return probe(x, training=False)
    def original():
        i = Interpreter(model_path=str(a.model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                        experimental_preserve_all_tensors=True, num_threads=1)
        i.allocate_tensors()
        return i
    template = original()
    details = {d['index']: d for d in template.get_tensor_details()}
    ops = [op for op in template._get_ops_details() if op['op_name'] in ('CONV_2D', 'DEPTHWISE_CONV_2D')]
    if len(ops) != len(tap_layers):
        raise ValueError('Intermediate operator count mismatch')
    inp, out = template.get_input_details()[0], template.get_output_details()[0]
    scale, zero = inp['quantization']; oscale, ozero = out['quantization']
    intermediate = [dict(name=l.name, original_op=op['op_name'], values=0,
                         absolute_error_sum=0., absolute_error_max=0., outside_range=0)
                    for l, op in zip(tap_layers, ops)]
    results, differences = [], []
    reset_checked = False
    for row in json.loads((a.audio/'manifest.json').read_text()):
        path = (a.audio/row['path']).resolve()
        if not path.is_relative_to(a.audio.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed probe WAV')
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or sha_pcm(pcm) != row['pcm_sha256']:
            raise ValueError('Changed probe PCM')
        raw = features(np.concatenate((np.zeros(32000, np.int16), pcm, np.zeros(16000, np.int16))))
        quantized = np.clip(np.rint(np.clip(raw.astype(np.float32)/25.6, 0, 26)/scale)+zero, -128, 127).astype(np.int8)
        def run(accumulate):
            i = original()
            for state in states:
                state.assign(tf.zeros_like(state))
            expected, actual = [], []
            for at in range(0, len(quantized)-2, 3):
                q = quantized[None, at:at+3]
                i.set_tensor(inp['index'], q); i.invoke()
                floats = infer((q.astype(np.float32)-zero)*scale)
                expected.append((int(i.get_tensor(out['index'])[0, 0])-ozero)*oscale)
                actual.append(float(floats[-1].numpy()[0, 0]))
                if accumulate:
                    for op, value, stats in zip(ops, floats[:-1], intermediate):
                        index = int(op['outputs'][0]); detail = details[index]
                        s, z = detail['quantization']
                        if s <= 0:
                            raise ValueError('Expected per-tensor activation quantization')
                        original_value = (i.get_tensor(index).astype(np.float32)-z)*s
                        f = value.numpy()
                        if f.shape != original_value.shape:
                            raise ValueError('Intermediate activation shape mismatch')
                        delta = np.abs(f-original_value)
                        bounds = np.iinfo(detail['dtype'])
                        stats['values'] += f.size
                        stats['absolute_error_sum'] += float(delta.sum())
                        stats['absolute_error_max'] = max(stats['absolute_error_max'], float(delta.max()))
                        stats['outside_range'] += int(((f < (bounds.min-z)*s) | (f > (bounds.max-z)*s)).sum())
                        stats['original_quantization_scale'] = float(s)
            return np.asarray(expected), np.asarray(actual)
        expected, actual = run(True)
        if not reset_checked and row['teacher_target_text']:
            e2, a2 = run(False)
            np.testing.assert_array_equal(e2, expected)
            np.testing.assert_allclose(a2, actual, rtol=0, atol=1e-7)
            reset_checked = True
        delta = np.abs(actual-expected); differences.extend(delta.tolist())
        results.append(dict(path=row['path'], text=row['text'], teacher_target_text=row['teacher_target_text'],
                            steps=len(actual), original_max=float(expected.max()), reconstructed_max=float(actual.max()),
                            mean_absolute_error=float(delta.mean()), max_absolute_error=float(delta.max())))
    for layer in intermediate:
        layer['mean_error_lsb'] = layer['absolute_error_sum']/layer['values']/layer['original_quantization_scale']
        layer['max_error_lsb'] = layer['absolute_error_max']/layer['original_quantization_scale']
        layer['outside_range_fraction'] = layer['outside_range']/layer['values']
    # Predeclared diagnostic gates. Passing permits investigation, not acceptance.
    high = sum(r['teacher_target_text'] and r['original_max'] >= .5 for r in results)
    passed = (high >= 4 and np.mean(differences) <= .01 and np.max(differences) <= .15
              and all(l['mean_error_lsb'] <= 5 for l in intermediate))
    report = dict(role='Teacher-target reconstruction and intermediate-activation diagnostic, not Nyx recognition evidence',
                  source_model_sha256=SOURCE_SHA, reconstruction_weights_sha256=sha(a.reconstruction/'reconstructed.weights.h5'),
                  source_sha256=sha(Path(__file__)), audio_generation_sha256=sha(a.audio/'generation.json'),
                  matched_input_quantization=True, reset_repeatability_passed=reset_checked,
                  high_response_target_clips=high, mean_absolute_error=float(np.mean(differences)),
                  max_absolute_error=float(np.max(differences)), intermediate_layers=intermediate, clips=results,
                  diagnostic_gate=dict(minimum_high_response_clips=4, maximum_mean_probability_error=.01,
                                       maximum_probability_error=.15, maximum_layer_mean_lsb_error=5, passed=bool(passed)),
                  validation_or_final_accessed=False,
                  limits='Small synthetic training-speaker probe; float dequantized reconstruction is not bit-exact INT8. Passing is not transfer efficacy, recognition acceptance, MCU certification or Hi ESP equivalence.')
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('clips', 'intermediate_layers')}))


def sha_pcm(pcm):
    return hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()


if __name__ == '__main__':
    main()
