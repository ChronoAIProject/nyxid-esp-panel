"""Training-only conditional likelihood of explicit finite CTC hypotheses."""
import numpy as np


def losses(logits, intent_ids, hypotheses, lengths):
    """Per-example cross-entropy over raw equal-prior CTC likelihoods."""
    import tensorflow as tf
    references = tf.convert_to_tensor(hypotheses, dtype=tf.int32)
    sizes = tf.convert_to_tensor(lengths, dtype=tf.int32)
    count, classes = tf.shape(logits)[0], tf.shape(references)[0]

    def nonempty():
        nll = tf.nn.ctc_loss(tf.tile(references, [count, 1]),
            tf.repeat(logits, classes, axis=0), tf.tile(sizes, [count]),
            tf.fill([count*classes], tf.shape(logits)[1]),
            logits_time_major=False, blank_index=0)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=intent_ids, logits=tf.reshape(-nll, [count, classes]))

    return tf.cond(count > 0, nonempty, lambda: tf.zeros([0], dtype=logits.dtype))


def compact_loss(teacher, logits):
    """Mean four-intent loss on original [hey, intent] paired targets only."""
    import tensorflow as tf
    from phrase_ctc_readout import targets
    labels = tf.argmax(teacher, axis=-1, output_type=tf.int32)
    tf.debugging.assert_equal(tf.shape(teacher)[-1], 6)
    tf.debugging.assert_equal(tf.shape(logits)[-1], 6)
    tf.debugging.assert_equal(labels[:, 0], tf.ones([tf.shape(labels)[0]], tf.int32))
    tf.debugging.assert_equal(labels[:, 2:], tf.zeros_like(labels[:, 2:]))
    intents = labels[:, 1]-2
    tf.debugging.assert_greater_equal(intents, 0)
    tf.debugging.assert_less(intents, 4)
    # Match the original dense target width, avoiding the known TF2.21
    # optimizer issue from combining differently padded CTC targets.
    references = tf.pad(tf.constant(targets(np.arange(4))),
        [[0, 0], [0, tf.shape(labels)[1]-2]])
    values = losses(logits, intents, references, [2]*4)
    return tf.cond(tf.shape(values)[0] > 0, lambda: tf.reduce_mean(values),
                   lambda: tf.zeros((), dtype=logits.dtype))
