"""Verify fixed-statistics affine adaptation using real training feature windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
from batchnorm_policy import apply_policy, snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'features', 'weights', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output report')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(a.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
    arch = ['--residual_connection', '0,0,0,0', '--stride', '3', '--pointwise_filters', '60,60,60,60',
            '--repeat_in_block', '1,1,1,1', '--mixconv_kernel_sizes', '[5], [9], [13], [21]',
            '--first_conv_filters', '30', '--first_conv_kernel_size', '5']
    model = mixednet.model(parser.parse_args(arch), (150, 40), None)
    head = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense))
    initial_head = head.get_weights()
    weights_sha = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    if weights_sha != '8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b':
        raise ValueError('Unexpected reconstructed weights')
    model.load_weights(a.weights); head.set_weights(initial_head)
    rows = json.loads((a.features/'features.json').read_text())
    selected = [next(row for row in rows if row['split'] == 'training' and row['label'] == label)
                for label in (0, 1)]
    raw = np.memmap(a.features/'features.bin', mode='r', dtype='<u2').reshape(-1,40)
    x = []
    for row in selected:
        end = max(150, min(row['frames'], row['keyword_frame']+8 if row['label'] else row['frames']-50))
        x.append(raw[row['offset']+end-150:row['offset']+end])
    x = np.clip(np.asarray(x, dtype=np.float32)/25.6, 0, 26)
    y = np.asarray([0., 1.], dtype=np.float32)[:, None]
    before = model(x, training=False).numpy()
    apply_policy(model, 'affine')
    np.testing.assert_array_equal(model(x, training=False).numpy(), before)
    np.testing.assert_array_equal(model(x, training=True).numpy(), before)
    initial = snapshot(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    for _ in range(3):
        model.train_on_batch(x, y)
    trained = snapshot(model)
    assert trained['moving_statistics_sha256'] == initial['moving_statistics_sha256']
    assert trained['affine_sha256'] != initial['affine_sha256']
    padded = np.pad(x[0], ((0,3),(0,0)))
    # Compute window reference before upstream conversion changes source stream modes.
    aligned = model(padded[None,1:151], training=False).numpy()
    streaming = utils.to_streaming_inference(model, {'stride':3, 'spectrogram_length':150},
                                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1,3,40), np.float32))
    for variable in streaming.variables:
        if 'states' in variable.name:
            variable.assign(tf.zeros_like(variable))
    values = [streaming(padded[None,at:at+3], training=False).numpy() for at in range(0,153,3)]
    np.testing.assert_allclose(values[49], aligned, rtol=0, atol=1e-6)
    result = dict(role='Training-only BN affine policy and streaming-clone diagnostic, not recognition evidence',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy_source_sha256=hashlib.sha256(Path(__file__).with_name('batchnorm_policy.py').read_bytes()).hexdigest(),
                  initialization_output_exactly_preserved=True, train_call_initial_output_exactly_preserved=True,
                  statistics_unchanged=True, affine_parameters_updated=True,
                  phase_aligned_streaming_error=float(np.max(np.abs(values[49]-aligned))),
                  initial=initial, trained=trained, training_paths=[row['path'] for row in selected],
                  weights_sha256=weights_sha, validation_or_final_accessed=False)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
