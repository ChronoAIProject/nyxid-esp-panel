"""Explicit empty-sequence supervision for existing ambient-negative draws."""
import numpy as np


def mark_ambient(labels, count):
    """Mark the final ambient cohort before the caller shuffles the batch."""
    y = np.asarray(labels)
    if (y.ndim != 2 or y.shape[1] != 2 or type(count) is not int
            or not 0 < count <= len(y) or not np.isfinite(y).all()
            or np.any(y[-count:] != 0)):
        raise ValueError('Ambient markers require nonwake, unpaired final rows')
    marker = np.zeros((len(y), 1), dtype=y.dtype)
    marker[-count:] = 1
    return np.concatenate((y, marker), axis=1)


def mask(labels):
    import tensorflow as tf
    tf.debugging.assert_rank(labels, 2)
    tf.debugging.assert_equal(tf.shape(labels)[1], 3)
    flags = labels[:, 2]
    tf.debugging.assert_all_finite(flags, 'Nonfinite background markers')
    tf.debugging.assert_equal(tf.reduce_all((flags == 0) | (flags == 1)), True)
    selected = flags == 1
    tf.debugging.assert_positive(tf.math.count_nonzero(selected))
    tf.debugging.assert_equal(tf.boolean_mask(labels[:, :2], selected), 0.)
    return selected


def empty_target_loss(logits):
    """Empty CTC has one all-blank path; normalize its NLL by frame count."""
    import tensorflow as tf
    tf.debugging.assert_rank(logits, 3)
    tf.debugging.assert_positive(tf.shape(logits)[0])
    tf.debugging.assert_positive(tf.shape(logits)[1])
    tf.debugging.assert_greater(tf.shape(logits)[2], 1)
    return -tf.reduce_mean(tf.nn.log_softmax(logits, axis=-1)[..., 0])
