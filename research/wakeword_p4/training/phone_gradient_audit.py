"""Read-only gradient decomposition on actual mixed-source training batches."""
import hashlib
from pathlib import Path

import numpy as np


def digest(variables):
    value = hashlib.sha256()
    for variable in variables:
        value.update(variable.numpy().tobytes())
    return value.hexdigest()


def batch_gradients(model, supervisor, targets, x, y, weights, gain):
    import tensorflow as tf

    full = supervisor.auxiliary_model
    with tf.GradientTape(persistent=True) as tape:
        probability, logits = full(x, training=True)
        binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], probability)*weights)
        ids = tf.cast(y[:, 1], tf.int32)
        present = ids > 0
        q = tf.gather(tf.constant(targets), tf.boolean_mask(ids, present)-1)
        phone = supervisor.phone_loss(q, tf.boolean_mask(logits, present))
        joint = binary+gain*phone
    variables = full.trainable_variables
    arrays = []
    for loss in (binary, phone, joint):
        values = tape.gradient(loss, variables)
        # Keras variables expose dtype names as strings; TensorFlow's
        # unconnected_gradients=ZERO path mishandles these in this runtime.
        arrays.append([np.asarray(g).ravel().astype(np.float64) if g is not None
                       else np.zeros(tuple(v.shape), np.float64).ravel()
                       for g, v in zip(values, variables, strict=True)])
    del tape
    classifier = {id(v) for v in model.get_layer('classifier').trainable_variables}
    head = {id(v) for v in full.get_layer('training_phone_logits').trainable_variables}
    groups = {'all': range(len(variables)),
              'encoder': [i for i, v in enumerate(variables) if id(v) not in classifier | head],
              'classifier': [i for i, v in enumerate(variables) if id(v) in classifier],
              'auxiliary_head': [i for i, v in enumerate(variables) if id(v) in head]}
    result = {'binary_loss': float(binary), 'phone_loss': float(phone),
              'joint_loss': float(joint), 'supervised_examples': int(np.count_nonzero(y[:, 1])),
              'groups': {}}
    for name, indices in groups.items():
        indices = list(indices)
        vectors = [np.concatenate([values[i] for i in indices]) for values in arrays]
        binary_g, phone_g, joint_g = vectors
        nb, np_, nj = [float(np.linalg.norm(v)) for v in vectors]
        residual = float(np.max(np.abs(joint_g-(binary_g+gain*phone_g))))
        if not all(np.isfinite(v).all() for v in vectors) or residual > 1e-5*max(1, nb, np_):
            raise ValueError('Nonfinite or inconsistent gradient decomposition')
        result['groups'][name] = {
            'binary_norm': nb, 'phone_norm': np_, 'joint_norm': nj,
            'scaled_phone_to_binary_norm': gain*np_/nb if nb else None,
            'cosine': float(np.dot(binary_g, phone_g)/(nb*np_)) if nb*np_ else None,
            'decomposition_max_error': residual}
    limit = float(model.optimizer.global_clipnorm)
    total = result['groups']['all']
    result['binary_clip_scale'] = min(1., limit/total['binary_norm']) if total['binary_norm'] else 1.
    result['joint_clip_scale'] = min(1., limit/total['joint_norm']) if total['joint_norm'] else 1.
    result['classifier_binary_scale_relative_to_binary_only'] = result['joint_clip_scale']/result['binary_clip_scale']
    return result


def audit(model, supervisor, targets, batches, gain, plan):
    checkpoint = Path(plan['auxiliary_checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['auxiliary_checkpoint_sha256']:
        raise ValueError('Auxiliary checkpoint identity differs')
    before_binary = digest(model.weights)
    supervisor.auxiliary_model.load_weights(checkpoint)
    if digest(model.weights) != before_binary:
        raise ValueError('Auxiliary checkpoint changes the selected binary state')
    before = digest(supervisor.auxiliary_model.weights)
    optimizer_before = digest(model.optimizer.variables)
    prefix, results = hashlib.sha256(), []
    for x, y, weights in batches:
        if len(x) != plan['batch_size'] or y.shape != (len(x), 2) or weights.shape != (len(x),):
            raise ValueError('Unexpected mixed-batch geometry')
        for value in (x, y, weights):
            prefix.update(value.astype('<f4', copy=False).tobytes())
        results.append(batch_gradients(model, supervisor, targets, x, y, weights, gain))
    if len(results) != plan['batches'] or prefix.hexdigest() != plan['expected_training_prefix_sha256']:
        raise ValueError('Original mixed training prefix does not reproduce')
    if digest(supervisor.auxiliary_model.weights) != before or digest(model.optimizer.variables) != optimizer_before:
        raise ValueError('Gradient observation changed weights or optimizer')
    return {
        'scope': __doc__, 'checkpoint_sha256': plan['auxiliary_checkpoint_sha256'],
        'training_prefix_sha256': prefix.hexdigest(), 'batch_count': len(results), 'gain': gain,
        'binary_weights_sha256': before_binary, 'full_weights_sha256': before,
        'weights_unchanged': True, 'optimizer_unchanged': True, 'batches': results,
        'limits': 'Actual first mixed training batches at one fixed saved state; no parameter/optimizer update or resume. These samples do not represent the complete training trajectory. Float64 norm/clip calculations diagnose the float32 optimizer approximately. No validation, model selection, inference export, room recording or board access.'}
