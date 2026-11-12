"""Training-only same-source rendering consistency; no deployment layers."""
import tensorflow as tf

EPSILON = 1e-8

def rendering_consistency(embeddings):
    """[2 IDs,4 phrases,5 arms,D]; quiet is arm0, all32 pairs equal."""
    tf.debugging.assert_rank(embeddings, 4)
    tf.debugging.assert_equal(tf.shape(embeddings)[:3], [2, 4, 5])
    tf.debugging.assert_all_finite(embeddings, 'Nonfinite classifier inputs')
    # sqrt(max(sum squares, eps²)) avoids evaluating a singular sqrt at zero.
    squared = tf.reduce_sum(tf.square(embeddings), axis=-1, keepdims=True)
    denominator = tf.sqrt(tf.maximum(squared, tf.constant(EPSILON**2, dtype=embeddings.dtype)))
    unit = embeddings/denominator
    cosine = tf.reduce_sum(unit[:, :, :1, :]*unit[:, :, 1:, :], axis=-1)
    return tf.reduce_mean(1-cosine)
