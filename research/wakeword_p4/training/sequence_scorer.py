"""Differentiable whole-sequence scorer sharing a frozen-BN window model.

The supported graph is a linear valid-convolution trunk with a final
Flatten/Dense sigmoid head, as in the teacher60 research model. This changes
neither its learned parameters nor the inference/export model. Source phase
and output-step accounting are the caller's responsibility.
"""
import tensorflow as tf


class SlidingDense(tf.keras.layers.Layer):
    """Apply the original Dense kernel to each contiguous trunk history."""
    def __init__(self, dense, history_shape, **kwargs):
        super().__init__(**kwargs)
        self.dense = dense
        self.history_shape = tuple(int(size) for size in history_shape)

    def call(self, inputs):
        kernel = tf.reshape(self.dense.kernel, (*self.history_shape, self.dense.units))
        logits = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='VALID')
        if self.dense.use_bias:
            logits = tf.nn.bias_add(logits, self.dense.bias)
        return tf.squeeze(self.dense.activation(logits), axis=(2, 3))


def build_sequence_scorer(window):
    """Reuse all original variables; return probabilities at the trunk stride.

    No model or optimizer update is performed. BN must remain frozen because
    a full sequence and a batch of overlapping windows have different batch
    statistics. The final convolution may use fewer frames than window.input
    because valid striding drops its unused last frames.
    """
    if (len(window.inputs) != 1 or len(window.outputs) != 1
            or len(window.input_shape) != 3 or window.input_shape[2] != 40):
        raise ValueError('Expected one [batch, frames, 40] feature input')
    flatten, dense = window.layers[-2:]
    if (not isinstance(flatten, tf.keras.layers.Flatten)
            or not isinstance(dense, tf.keras.layers.Dense) or dense.units != 1
            or dense.get_config()['activation'] != 'sigmoid'):
        raise ValueError('Expected final Flatten / single sigmoid Dense head')
    history = tuple(flatten.input.shape[1:])
    if len(history) != 3 or None in history or history[1] != 1:
        raise ValueError('Expected fixed [time, 1, channels] head history')
    if any(isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
           for layer in window.layers):
        raise ValueError('Whole-sequence scorer requires frozen batch normalization')
    if any(getattr(layer, 'mode', 'TRAINING') not in ('TRAINING', 'NON_STREAM_INFERENCE')
           for layer in window.layers):
        raise ValueError('Build from the window graph before any upstream streaming conversion')
    # Reusing layers, rather than copying their current weights, keeps gradients
    # attached to the original checkpoint's exact trainable variables.
    trunk = tf.keras.Model(window.input, flatten.input)
    dynamic_input = tf.keras.Input(shape=(None, 40), dtype=window.inputs[0].dtype)
    dynamic_trunk = tf.keras.models.clone_model(
        trunk, input_tensors=dynamic_input, clone_function=lambda layer: layer)
    output = SlidingDense(dense, history, name='shared_sliding_dense')(dynamic_trunk.output)
    sequence = tf.keras.Model(dynamic_input, output)
    if ({id(v) for v in sequence.variables} != {id(v) for v in window.variables}
            or {id(v) for v in sequence.trainable_variables}
            != {id(v) for v in window.trainable_variables}):
        raise ValueError('Sequence graph does not share all original variables and trainability')
    return sequence
