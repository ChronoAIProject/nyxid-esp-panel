"""Training-only absolute five-frame BCE tail risk; deployment graph unchanged."""
import tensorflow as tf


def temporal_tail_risk(logits, labels, weights):
    """Mean weighted log-mean-exp of five individual label-directed BCEs."""
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(tf.reshape(labels, [-1]), logits.dtype)
    weights = tf.cast(tf.reshape(weights, [-1]), logits.dtype)
    tf.debugging.assert_rank(logits, 2)
    tf.debugging.assert_equal(tf.shape(logits)[1], 5)
    tf.debugging.assert_equal(tf.shape(logits)[0], tf.size(labels))
    tf.debugging.assert_equal(tf.size(labels), tf.size(weights))
    tf.debugging.assert_positive(tf.size(labels))
    tf.debugging.assert_equal(tf.reduce_all((labels == 0) | (labels == 1)), True)
    tf.debugging.assert_non_negative(weights)
    for value in (logits, labels, weights):
        tf.debugging.assert_all_finite(value, 'Nonfinite temporal risk input')
    bce = tf.nn.softplus((1 - 2 * labels[:, None]) * logits)
    risk = tf.reduce_logsumexp(bce, axis=1) - tf.math.log(tf.cast(5, logits.dtype))
    return tf.reduce_mean(weights * risk)
