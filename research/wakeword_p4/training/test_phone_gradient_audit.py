"""Check diagnostic gradients against a closed-form scalar two-task model."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from phone_gradient_audit import audit, batch_gradients


class GradientAuditTest(unittest.TestCase):
    def example(self, target=10.):
        import tensorflow as tf
        x = tf.keras.Input((1,))
        z = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones')(x)
        probability = tf.keras.layers.Dense(1, use_bias=False, activation='sigmoid',
                                           kernel_initializer='ones', name='classifier')(z)
        phone = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones',
                                     name='training_phone_logits')(z)
        model = tf.keras.Model(x, probability)
        model.compile(optimizer=tf.keras.optimizers.Adam(.0002, global_clipnorm=5),
                      loss='binary_crossentropy')
        supervisor = SimpleNamespace(auxiliary_model=tf.keras.Model(x, (probability, phone)),
            phone_loss=lambda q, p: tf.reduce_mean(tf.square(p-q)))
        values = np.array([[1.], [2.]], np.float32)
        labels = np.array([[1., 1.], [0., 2.]], np.float32)
        weights = np.array([1., 5.], np.float32)
        return model, supervisor, np.full((2, 1), target, np.float32), values, labels, weights

    def test_conflicting_and_aligned_gradients_match_closed_form(self):
        binary = ((1/(1+math.exp(-1))-1)+10/(1+math.exp(-2)))/2
        for target, phone in ((10., -25.), (1., 2.)):
            with self.subTest(target=target):
                model, supervisor, q, x, y, w = self.example(target)
                before = [v.numpy().copy() for v in supervisor.auxiliary_model.weights]
                r = batch_gradients(model, supervisor, q, x, y, w, .1)
                e = r['groups']['encoder']
                self.assertAlmostEqual(e['binary_norm'], binary, places=5)
                self.assertAlmostEqual(e['phone_norm'], abs(phone), places=5)
                self.assertAlmostEqual(e['joint_norm'], abs(binary+.1*phone), places=5)
                self.assertAlmostEqual(e['cosine'], math.copysign(1., phone), places=6)
                expected_norm = math.sqrt((binary+.1*phone)**2+binary**2+(.1*phone)**2)
                self.assertAlmostEqual(r['groups']['all']['joint_norm'], expected_norm, places=5)
                self.assertAlmostEqual(r['joint_clip_scale'], min(1, 5/expected_norm), places=6)
                self.assertEqual(r['groups']['classifier']['phone_norm'], 0)
                self.assertEqual(r['groups']['auxiliary_head']['binary_norm'], 0)
                for old, variable in zip(before, supervisor.auxiliary_model.weights, strict=True):
                    np.testing.assert_array_equal(old, variable.numpy())
                self.assertEqual(int(model.optimizer.iterations), 0)

    def test_checkpoint_audit_preserves_state_and_rejects_prefix_tampering(self):
        model, supervisor, q, x, y, w = self.example()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'head.weights.h5'
            supervisor.auxiliary_model.save_weights(checkpoint)
            h = hashlib.sha256()
            for value in (x, y, w):
                h.update(value.astype('<f4').tobytes())
            plan = {'auxiliary_checkpoint': str(checkpoint),
                    'auxiliary_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    'batch_size': 2, 'batches': 1, 'expected_training_prefix_sha256': h.hexdigest()}
            result = audit(model, supervisor, q, [(x, y, w)], .1, plan)
            self.assertTrue(result['weights_unchanged'])
            self.assertTrue(result['optimizer_unchanged'])
            plan['expected_training_prefix_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'prefix'):
                audit(model, supervisor, q, [(x, y, w)], .1, plan)
            self.assertEqual(int(model.optimizer.iterations), 0)


if __name__ == '__main__':
    unittest.main()
