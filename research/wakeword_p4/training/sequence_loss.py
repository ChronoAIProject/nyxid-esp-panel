"""Opt-in, stable training losses for the shared whole-sequence scorer.

The original window/inference model and probability scorer stay unchanged.
Export a separate original window copy, never this training-only logits graph.
"""
import tensorflow as tf

from sequence_scorer import build_sequence_scorer


class SlidingLogits(tf.keras.layers.Layer):
    def __init__(self, probability_head, **kwargs):
        super().__init__(**kwargs)
        self.dense = probability_head.dense
        self.history_shape = probability_head.history_shape

    def call(self, inputs):
        kernel = tf.reshape(self.dense.kernel, (*self.history_shape, self.dense.units))
        logits = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='VALID')
        if self.dense.use_bias:
            logits = tf.nn.bias_add(logits, self.dense.bias)
        return tf.squeeze(logits, axis=(2, 3))


def build_sequence_logits(window):
    """Share all original variables, returning [batch, steps] pre-sigmoid values."""
    probabilities = build_sequence_scorer(window)
    head = probabilities.layers[-1]
    logits = SlidingLogits(head, name='shared_sliding_logits')(head.input)
    model = tf.keras.Model(probabilities.input, logits)
    if ({id(v) for v in model.variables} != {id(v) for v in window.variables}
            or {id(v) for v in model.trainable_variables}
            != {id(v) for v in window.trainable_variables}):
        raise ValueError('Logits graph must share all original variables')
    return model


def mean_probability_bce(labels, logits):
    """Per-example BCE of mean(sigmoid(logits)), without probability clipping.

This is the same mathematical mean-of-probabilities objective. It is not BCE
of mean logits or a mean of individual BCEs. Terminal BCE uses logits[:, -1:].
Binary or soft labels have shape [batch]; logits have shape [batch, steps].
Class/sample weights and batch reduction belong to the caller.
"""
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(labels, logits.dtype)
    tf.debugging.assert_rank(logits, 2)
    tf.debugging.assert_rank(labels, 1)
    tf.debugging.assert_equal(tf.shape(labels)[0], tf.shape(logits)[0])
    tf.debugging.assert_positive(tf.shape(logits)[1])
    count = tf.math.log(tf.cast(tf.shape(logits)[1], logits.dtype))
    positive = count-tf.reduce_logsumexp(-tf.nn.softplus(-logits), axis=1)
    negative = count-tf.reduce_logsumexp(-tf.nn.softplus(logits), axis=1)
    return labels*positive+(1-labels)*negative
