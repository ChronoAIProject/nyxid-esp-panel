"""Competing phrase classes with the original binary target probability at init."""
import numpy as np
import tensorflow as tf

CLASSES = ('other', 'hey nyx', 'hey mix', 'hey next', 'hey nick')


def expand_binary(model):
    old = model.get_layer('classifier')
    kernel, bias = old.get_weights()
    head = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='phrase_classifier',
                                 kernel_initializer='zeros', bias_initializer='zeros')
    probabilities = head(old.input)
    full = tf.keras.Model(model.input, probabilities)
    weights = np.zeros((kernel.shape[0], len(CLASSES)), dtype=np.float32)
    weights[:, 1:2] = kernel
    offsets = np.full(len(CLASSES), -np.log(len(CLASSES)-1), dtype=np.float32)
    offsets[1] = bias[0]
    head.set_weights((weights, offsets))
    # Sum exp(non-target logits) is one, so p(target) starts as sigmoid(old logit).
    # A fixed one-hot Dense keeps a single scalar output and standard TFLite ops.
    target = tf.keras.layers.Dense(1, use_bias=False, trainable=False, name='target_probability',
                                   kernel_initializer='zeros')
    scalar = tf.keras.Model(model.input, target(probabilities))
    selector = np.zeros((len(CLASSES), 1), dtype=np.float32)
    selector[1] = 1
    target.set_weights((selector,))
    return scalar, full
