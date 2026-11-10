"""Function-preserving 2x channel widening of the audited reconstructed teacher.

Net2Wider-style replication is a capacity experiment, not Nyx quality evidence.
Uses only dedicated compatibility probe WAVs, never recognition evaluation sets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import soundfile as sf
import tensorflow as tf
from features import features

ARCH = ['--residual_connection', '0,0,0,0', '--stride', '3',
        '--pointwise_filters', '60,60,60,60', '--repeat_in_block', '1,1,1,1',
        '--mixconv_kernel_sizes', '[5], [9], [13], [21]',
        '--first_conv_filters', '30', '--first_conv_kernel_size', '5']
WEIGHTS_SHA = '8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'weights', 'probe-audio', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    if sha(a.weights) != WEIGHTS_SHA:
        raise ValueError('Unexpected reconstructed source weights')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    sys.path.insert(0, str(a.upstream.resolve()))
    from microwakeword import mixednet
    parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
    wide_arch = [s.replace('60,60,60,60', '120,120,120,120') for s in ARCH]
    wide_arch[wide_arch.index('--first_conv_filters')+1] = '60'
    teacher = mixednet.model(parser.parse_args(ARCH), (150, 40), None)
    wide = mixednet.model(parser.parse_args(wide_arch), (150, 40), None)
    teacher.load_weights(a.weights)
    if len(teacher.layers) != len(wide.layers):
        raise ValueError('Layer topology changed')
    mapped = []
    for source_layer, target_layer in zip(teacher.layers, wide.layers):
        source = getattr(source_layer, 'cell', source_layer)
        target = getattr(target_layer, 'cell', target_layer)
        weights = source.get_weights()
        if type(source) is not type(target):
            raise ValueError('Operator type changed')
        if isinstance(source, tf.keras.layers.DepthwiseConv2D):
            converted = [np.repeat(weights[0], 2, axis=2)]
            if source.use_bias:
                converted.append(np.repeat(weights[1], 2))
        elif isinstance(source, tf.keras.layers.Conv2D):
            kernel = np.repeat(weights[0], 2, axis=3)
            if target.get_weights()[0].shape[2] == weights[0].shape[2]*2:
                kernel = np.repeat(kernel, 2, axis=2)/2
            elif target.get_weights()[0].shape[2] != weights[0].shape[2]:
                raise ValueError('Unexpected input channel expansion')
            converted = [kernel]
            if source.use_bias:
                converted.append(np.repeat(weights[1], 2))
        elif isinstance(source, tf.keras.layers.Dense):
            converted = [np.repeat(weights[0], 2, axis=0)/2] + weights[1:]
        elif isinstance(source, tf.keras.layers.BatchNormalization):
            converted = [np.repeat(w, 2) for w in weights]
            source.trainable = False; target.trainable = False
        else:
            if weights:
                raise ValueError(f'Unhandled weighted operator {source.name}')
            continue
        if [w.shape for w in converted] != [w.shape for w in target.get_weights()]:
            raise ValueError(f'Unexpected widened shapes in {source_layer.name}: {[w.shape for w in converted]} vs {[w.shape for w in target.get_weights()]}')
        target.set_weights(converted)
        mapped.append(dict(source=source_layer.name, target=target_layer.name,
                           source_shapes=[list(w.shape) for w in weights],
                           target_shapes=[list(w.shape) for w in converted]))
    rows = json.loads((a.probe_audio/'manifest.json').read_text())
    if len(rows) != 16 or not all('teacher_target_text' in r for r in rows):
        raise ValueError('Expected dedicated reconstruction probes only')
    results, errors = [], []
    for row in rows:
        path = a.probe_audio/row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Probe WAV changed')
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1:
            raise ValueError('Unexpected probe format')
        raw = features(np.pad(pcm, (32000, 16000)))
        x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        windows = np.lib.stride_tricks.sliding_window_view(x, 150, axis=0)[1::3].transpose(0, 2, 1)
        before, after = [], []
        for at in range(0, len(windows), 32):
            before.extend(teacher(windows[at:at+32], training=False).numpy().reshape(-1))
            after.extend(wide(windows[at:at+32], training=False).numpy().reshape(-1))
        difference = np.abs(np.asarray(before)-after)
        if not np.isfinite(difference).all() or difference.max() > 1e-5:
            raise ValueError('Function-preserving widening check failed')
        errors.extend(difference.tolist())
        results.append(dict(path=row['path'], teacher_target_text=row['teacher_target_text'],
                            teacher_max=float(max(before)), widened_max=float(max(after)),
                            max_absolute_error=float(difference.max())))
    if sum(r['teacher_target_text'] and r['teacher_max'] >= .5 for r in results) < 4:
        raise ValueError('Insufficient high-response probe coverage')
    a.output.mkdir(parents=True)
    wide.save_weights(a.output/'widened.weights.h5')
    report = dict(role='Function-preserving capacity expansion diagnostic; not Nyx recognition evidence',
                  source_weights_sha256=WEIGHTS_SHA, source_sha256=sha(Path(__file__)),
                  widened_weights_sha256=sha(a.output/'widened.weights.h5'),
                  probe_manifest_sha256=sha(a.probe_audio/'manifest.json'),
                  teacher_parameters=teacher.count_params(), widened_parameters=wide.count_params(),
                  architecture_args=wide_arch, context_frames=150, mapped_layers=mapped,
                  mean_absolute_error=float(np.mean(errors)), max_absolute_error=float(max(errors)),
                  passed=True, clips=results, validation_or_final_accessed=False,
                  reference='https://arxiv.org/abs/1511.05641',
                  limits='Duplicate channels preserve initial teacher function in floating point on these probes. Original BN statistics remain unavailable. No fine tuning, INT8 export, streaming parity, device resources or Nyx quality is established; Nyx training must reset classifier head.')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('mapped_layers','clips')}, indent=2))


if __name__ == '__main__':
    main()
