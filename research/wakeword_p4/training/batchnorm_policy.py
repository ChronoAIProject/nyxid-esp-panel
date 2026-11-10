"""Keep reconstructed BN statistics fixed while optionally learning its affine map.

Only the training call is overridden. Standard BatchNormalization inference and
weight serialization remain unchanged, so upstream streaming cloning uses the
same learned affine map with the unchanged moving statistics.
"""
import hashlib
from types import MethodType
import tensorflow as tf


def _fixed_statistics_call(self, inputs, training=None, mask=None):
    return tf.keras.layers.BatchNormalization.call(self, inputs, training=False, mask=mask)


def apply_policy(model, policy):
    if policy not in ('train', 'frozen', 'affine'):
        raise ValueError('Unknown batch normalization policy')
    layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    if policy != 'train' and not layers:
        raise ValueError('No batch normalization layers found')
    for layer in layers:
        layer.trainable = policy != 'frozen'
        if policy == 'affine':
            layer.call = MethodType(_fixed_statistics_call, layer)
    return [layer.name for layer in layers]


def snapshot(model):
    layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    def fingerprint(attributes):
        digest = hashlib.sha256()
        for layer in layers:
            for attribute in attributes:
                value = getattr(layer, attribute)
                if value is not None:
                    digest.update(value.numpy().tobytes())
        return digest.hexdigest()
    return dict(layers=[layer.name for layer in layers],
                moving_statistics_sha256=fingerprint(('moving_mean', 'moving_variance')),
                affine_sha256=fingerprint(('gamma', 'beta')))
