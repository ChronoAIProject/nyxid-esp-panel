"""Use the unchanged fitting pipeline with one opt-in research topology.

Keep train_head.py byte-identical so existing hash-bound experiments remain
replayable. Only this entry point opts into the ordered Dense32 architecture.
"""
import train_head

BASE_MAKE_MODEL = train_head.make_model


def make_model(architecture='conv24'):
    if architecture != 'temporal32-bn-dense32-dropout':
        return BASE_MAKE_MODEL(architecture)
    tf = train_head.tf
    x = tf.keras.Input(shape=(16, 96, 1), name='embedding_history')
    y = tf.keras.layers.Conv2D(32, (5, 96), use_bias=False)(x)
    y = tf.keras.layers.BatchNormalization(epsilon=1e-3)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(32, (5, 1), use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(epsilon=1e-3)(y)
    y = tf.keras.layers.ReLU()(y)
    # All eight positions remain ordered; trainable cross-channel combinations
    # can represent a conjunction of the two words.
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(32, activation='relu')(y)
    y = tf.keras.layers.Dropout(.15)(y)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(y)
    return tf.keras.Model(x, y)


if __name__ == '__main__':
    train_head.make_model = make_model
    train_head.main()
