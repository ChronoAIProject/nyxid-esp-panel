"""Measure potential teacher60 frame support on an all-positive diagnostic graph.

No learned checkpoint, PCM, feature inference dataset, classifier decision,
training or device is used. Positive normalized weights keep every possible
path active, allowing the logits/input Jacobian to expose structural support.
This does not measure a trained model's actual dependence on source phonemes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf

from probe_sequence_scorer import ARCH, UPSTREAM
from sequence_loss import build_sequence_logits


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'pack-records', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip() == UPSTREAM
    record_bytes = args.pack_records.read_bytes()
    assert sha(record_bytes) == '757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
    records = json.loads(record_bytes)
    assert len(records) == 7782
    args.output.mkdir(parents=True, exist_ok=False)
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    parser = argparse.ArgumentParser()
    mixednet.model_parameters(parser)
    model = mixednet.model(parser.parse_args(ARCH), (150, 40), None)
    for variable in model.variables:
        path, shape = variable.path, tuple(variable.shape)
        if path.endswith('/kernel'):
            divisor = np.prod(shape[:2]) if 'depthwise' in path else np.prod(shape[:-1])
            variable.assign(tf.ones_like(variable)/int(divisor))
        elif path.endswith('/gamma') or path.endswith('/moving_variance'):
            variable.assign(tf.ones_like(variable))
        elif any(path.endswith('/'+name) for name in ('bias', 'beta', 'moving_mean')):
            variable.assign(tf.zeros_like(variable))
        else:
            raise ValueError('Unknown graph variable: '+path)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    logits = build_sequence_logits(model)
    data = tf.ones((1, 162, 40), dtype=tf.float32)
    before = [v.numpy().copy() for v in model.variables]
    with tf.GradientTape() as tape:
        tape.watch(data)
        value = logits(data, training=False)[0]
    jacobian = tape.jacobian(value, data).numpy()[:, 0]
    assert value.shape == (5,) and jacobian.shape == (5, 162, 40)
    assert np.isfinite(jacobian).all()
    support = []
    for i, gradient in enumerate(jacobian):
        used = np.flatnonzero(np.any(gradient != 0, axis=1))
        np.testing.assert_array_equal(used, np.arange(i*3, i*3+149))
        assert np.all(gradient[used] > 0)
        support.append({'score_index': i, 'first_frame': int(used[0]),
            'last_frame_exclusive': int(used[-1])+1, 'used_frame_count': len(used),
            'minimum_positive_input_gradient': float(np.min(gradient[used])),
            'maximum_input_gradient': float(np.max(gradient)),
            'unused_input_frames': np.flatnonzero(np.all(gradient == 0, axis=1)).tolist()})
    changed = data.numpy();changed[:, 161] = 100
    np.testing.assert_array_equal(logits(changed, training=False).numpy()[0], value.numpy())
    # The first Conv5/stride3 consumes0..148; subsequent valid depthwise
    # convolutions and the Dense head combine all49 outputs. Input149 is unused.
    first_conv_outputs = (150-5)//3+1
    analytic_frames = (first_conv_outputs-1)*3+5
    assert first_conv_outputs == 49 and analytic_frames == 149
    # Pinned MicroFrontend's public framing contract: 30ms windows / 10ms hops.
    # Verify initial feature emission rather than inferring it from dataset ends.
    from pymicro_features import MicroFrontend
    frontend = MicroFrontend();emission = []
    for hop in range(1, 11):
        frame = frontend.process_samples(np.zeros(160, dtype=np.int16).tobytes())
        assert frame.samples_read == 160
        emission.append({'hop': hop, 'feature_emitted': bool(frame.features)})
    assert [r['hop'] for r in emission if r['feature_emitted']] == list(range(3, 11))
    for row in records:
        begin = row['feature_crop_end']-150
        for s, saved in zip(support, row['five_window_sample_bounds'], strict=True):
            # Union of149 consecutive 30ms frames, excluding unused crop tail.
            actual = [(begin+s['first_frame'])*160,
                      (begin+s['last_frame_exclusive']+2)*160]
            assert actual == saved
            assert actual[0] <= 32000 and actual[1] >= 32000+row['signal_samples']
    assert all(np.array_equal(v.numpy(), initial) for v, initial in zip(model.variables, before, strict=True))
    arrays = args.output/'jacobian.npz'
    np.savez_compressed(arrays, input=data.numpy(), logits=value.numpy(), jacobian=jacobian)
    result = {'scope': __doc__, 'source_sha256': sha(Path(__file__).read_bytes()),
        'upstream_revision': UPSTREAM, 'architecture_args': ARCH, 'parameters': model.count_params(),
        'tensorflow': tf.__version__, 'pack_records_sha256': sha(record_bytes),
        'input_window_frames': 150, 'analytic_first_conv_outputs': first_conv_outputs,
        'potential_effective_frames': analytic_frames,
        'support': support, 'frontend_emission': emission,
        'all_candidate_bounds_match_measured_frame_support': True,
        'candidate_variants_checked': len(records), 'candidate_bounds_checked': len(records)*5,
        'unused_final_sequence_frame_perturbation_exactly_invariant': True,
        'variables_unchanged_after_diagnostic_assignment': True, 'optimizer_steps': 0,
        'arrays_sha256': sha(arrays.read_bytes()),
        'limits': ['Assigned diagnostic weights are not a learned checkpoint or candidate, and are not exported for deployment.',
            'Potential model frame support agrees with the existing conservative sample ledger; no crop repair or shifted data is introduced.',
            'Sample interval mapping uses the pinned30ms/10ms frontend framing contract. Emission timing checks alone are not a frontend dependency proof.',
            'Frontend adaptive state may depend on older PCM outside nominal windows. No claim that these are the only influential PCM samples.',
            'No source pronunciation, learned-model sensitivity, streaming score, cross-host numerical, recognition, P4 or Hi ESP acceptance evidence.']}
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
