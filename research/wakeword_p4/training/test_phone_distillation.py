"""Check phone timestamps, source isolation, sampler labels and control loss."""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phone_distillation import Sampler, supervisor
from prepare_phone_bank import FRAMES, STEPS, VOCAB, align_posteriors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhonePreparationTest(unittest.TestCase):
    def test_alignment_uses_only_frames_in_each_trailing_interval(self):
        source = np.zeros((168, VOCAB), np.float32)
        source[np.arange(168), np.arange(168)] = 1
        result = align_posteriors(source)
        self.assertEqual(result.shape, (STEPS, VOCAB))
        for i, row in enumerate(result):
            end = (139+3*i)*160
            indices = np.flatnonzero(row)
            self.assertTrue(np.all(400+320*indices <= end))
            self.assertTrue(np.all(400+320*indices > end-480))
            self.assertAlmostEqual(float(row.sum()), 1.)
        with self.assertRaises(ValueError):
            align_posteriors(source[:-1])

    def test_bank_retains_binary_roles_and_rejects_reserved_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            x = np.stack([np.full((FRAMES, 40), i, np.uint16) for i in range(4)])
            targets = np.full((4, STEPS, VOCAB), 1/VOCAB, np.float32)
            np.savez_compressed(path/'bank.npz', features=x, posteriors=targets)
            rows = [{'path': f'{i}.wav', 'variant': 0, 'label': 1-i%2, 'pair': i//2,
                     'source_sha256': str(i), 'feature_sha256': hashlib.sha256(x[i].tobytes()).hexdigest()}
                    for i in range(4)]
            meta = {'feature_index_sha256': 'fixed-index', 'frames': FRAMES, 'encoder_steps': STEPS,
                    'vocabulary': VOCAB, 'blank': 0, 'pair_count': 2, 'rows': rows,
                    'data_sha256': digest(path/'bank.npz')}
            (path/'bank.json').write_text(json.dumps(meta))
            original = [{'path': r['path'], 'variant': 0, 'split': 'training',
                         'sha256': r['source_sha256'], 'label': r['label']} for r in rows]
            sampler = Sampler(path, original, 'fixed-index', FRAMES)
            data = np.zeros((128, FRAMES, 40), np.uint16)
            ids = sampler.apply(data)[:, 0].astype(int)
            for i in range(8):
                positive, negative = ids[i]-1, ids[32+i]-1
                self.assertEqual(rows[positive]['label'], 1)
                self.assertEqual(rows[negative]['label'], 0)
                np.testing.assert_array_equal(data[i], x[positive])
                np.testing.assert_array_equal(data[32+i], x[negative])
            self.assertEqual(sampler.metadata()['draws'], 8)
            with self.assertRaises(ValueError):
                Sampler(path, original, 'fixed-index', FRAMES, 'ctc')
            teacher = np.zeros((4, 168, VOCAB), np.float32)
            for i in range(4):
                teacher[i, :, i+1] = 1
            aligned = np.stack([align_posteriors(row) for row in teacher])
            np.savez_compressed(path/'bank.npz', features=x, posteriors=aligned, teacher_posteriors=teacher)
            meta.update(retained_teacher_frames=168, teacher_frame_stride_samples=320,
                        teacher_frame_support_samples=400, data_sha256=digest(path/'bank.npz'))
            (path/'bank.json').write_text(json.dumps(meta))
            ctc = Sampler(path, original, 'fixed-index', FRAMES, 'ctc')
            np.testing.assert_array_equal(ctc.targets[:, 0].argmax(-1), [1, 2, 3, 4])
            self.assertTrue((ctc.targets[:, 1:].argmax(-1) == 0).all())
            teacher[0, -1, 1] = 0; teacher[0, -1, 2] = 1
            np.savez_compressed(path/'bank.npz', features=x, posteriors=aligned, teacher_posteriors=teacher)
            meta['data_sha256'] = digest(path/'bank.npz')
            (path/'bank.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                Sampler(path, original, 'fixed-index', FRAMES, 'ctc')
            original[0]['split'] = 'validation'
            with self.assertRaises(ValueError):
                Sampler(path, original, 'fixed-index', FRAMES)

    def test_zero_gain_matches_weighted_binary_optimizer_step(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(37)
        def make():
            x = tf.keras.Input((STEPS*60,))
            model = tf.keras.Model(x, tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x))
            model.compile(optimizer=tf.keras.optimizers.Adam(5e-5, global_clipnorm=5),
                          loss='binary_crossentropy')
            return model
        rng = np.random.default_rng(44)
        x = rng.normal(0, .05, (4, STEPS*60)).astype(np.float32)
        y = np.array([[1.], [0.], [1.], [0.]], np.float32)
        weights = np.array([1., 5., 1., 5.], np.float32)
        q = np.full((4, STEPS, VOCAB), 1/VOCAB, np.float32)
        ids = np.arange(1, 5, dtype=np.float32)[:, None]
        for objective in ('joint', 'conditional', 'ctc'):
            with self.subTest(objective=objective):
                control, auxiliary = make(), make()
                auxiliary.set_weights(control.get_weights())
                targets = q.copy()
                if objective == 'ctc':
                    targets[:] = 0
                    targets[..., 0] = 1
                    targets[:, 0, 0] = 0
                    targets[:, 0, 1] = 1
                train = supervisor(auxiliary, targets, 0., objective)
                train(x, np.concatenate((y, ids), axis=1), weights)
                control.train_on_batch(x, y, sample_weight=weights)
                for a, b in zip(control.get_weights(), auxiliary.get_weights(), strict=True):
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)

    def test_warmup_fits_only_head_and_zero_gain_still_matches_binary_update(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(37)

        def make():
            x = tf.keras.Input((STEPS, 60))
            z = tf.keras.layers.Dense(60, use_bias=False, kernel_initializer='identity')(x)
            probability = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(
                tf.keras.layers.Flatten()(z))
            model = tf.keras.Model(x, probability)
            model.compile(optimizer=tf.keras.optimizers.Adam(5e-5, global_clipnorm=5),
                          loss='binary_crossentropy')
            return model

        control, auxiliary = make(), make()
        auxiliary.set_weights(control.get_weights())
        labels = (np.arange(4)[:, None]+np.arange(STEPS)[None, :]) % 4
        features = np.eye(60, dtype=np.uint16)[labels]*256
        targets = np.eye(VOCAB, dtype=np.float32)[labels]
        train = supervisor(auxiliary, targets, 0.)
        before = auxiliary.get_weights()
        report = train.warmup(features, steps=25, learning_rate=.02, batch_size=4)
        for a, b in zip(before, auxiliary.get_weights(), strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(int(auxiliary.optimizer.iterations), 0)
        self.assertEqual(report['binary_weights_before_sha256'], report['binary_weights_after_sha256'])
        self.assertEqual(report['binary_optimizer_before_sha256'], report['binary_optimizer_after_sha256'])
        self.assertLess(report['history'][-1]['fit']['weighted_kl'], report['initial_fit']['weighted_kl']*.5)
        self.assertEqual(report['history'][-1]['fit']['nonblank_correct'], int((labels != 0).sum()))
        y = np.array([[1.], [0.], [1.], [0.]], np.float32)
        ids = np.arange(1, 5, dtype=np.float32)[:, None]
        weights = np.array([1., 5., 1., 5.], np.float32)
        x = features.astype(np.float32)/25.6
        train(x, np.concatenate((y, ids), axis=1), weights)
        control.train_on_batch(x, y, sample_weight=weights)
        for a, b in zip(control.get_weights(), auxiliary.get_weights(), strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        for steps, rate in ((0, .02), (1, float('nan'))):
            with self.assertRaises(ValueError):
                train.warmup(features, steps=steps, learning_rate=rate)

    def test_conditional_loss_excludes_ctc_blank_and_uses_teacher_phone_mass(self):
        import tensorflow as tf
        x = tf.keras.Input((STEPS*60,))
        model = tf.keras.Model(x, tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x))
        q = np.zeros((1, 2, VOCAB), np.float32)
        q[0, 0, [0, 1, 2]] = [.9, .075, .025]
        q[0, 1, [0, 1, 2]] = [.2, .2, .6]
        # Explicit 391-way uniform conditional predictor; changing its blank
        # logit must have no effect. Compute the weighted reference by hand.
        phones = np.zeros_like(q)
        train = supervisor(model, q, .01, 'conditional')
        joint = supervisor(model, q, .01, 'joint')
        reference_joint = sum(w*sum(t*np.log(t*392) for t in distribution)
                              for w, distribution in ((.145, (.9, .075, .025)),
                                                      (.81, (.2, .2, .6))))/.955
        self.assertAlmostEqual(float(joint.phone_loss(q, phones)), reference_joint, places=5)
        reference = sum(w*sum(t*np.log(t*391) for t in distribution)
                        for w, distribution in ((.1, (.75, .25)), (.8, (.25, .75))))/.9
        self.assertAlmostEqual(float(train.phone_loss(q, phones)), reference, places=5)
        phones[..., 0] = 100
        self.assertAlmostEqual(float(train.phone_loss(q, phones)), reference, places=5)
        blank = np.zeros_like(q); blank[..., 0] = 1
        self.assertEqual(float(train.phone_loss(blank, phones)), 0.)
        with tf.GradientTape() as tape:
            logits = tf.Variable(phones)
            loss = train.phone_loss(q, logits)
        gradient = tape.gradient(loss, logits).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        np.testing.assert_array_equal(gradient[..., 0], 0)

    def test_ctc_loss_matches_exhaustive_paths_with_repeated_labels(self):
        import tensorflow as tf
        from phone_sequences import collapse
        x = tf.keras.Input((STEPS*60,))
        model = tf.keras.Model(x, tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x))
        train = supervisor(model, np.zeros((1, STEPS, VOCAB), np.float32), .01, 'ctc')
        logits = np.array([[[.4, .1, -.2], [-.3, .2, .7], [.1, .9, .2]]], np.float32)
        probabilities = np.asarray(tf.nn.softmax(logits))[0]
        for sequence in ([1, 1], [1, 2], [2]):
            target = np.zeros((1, 3, 3), np.float32); target[..., 0] = 1
            for i, token in enumerate(sequence):
                target[0, i, 0] = 0; target[0, i, token] = 1
            total = sum(np.prod([probabilities[i, token] for i, token in enumerate(path)])
                        for path in itertools.product(range(3), repeat=3) if collapse(path) == sequence)
            self.assertAlmostEqual(float(train.phone_loss(target, logits)), -np.log(total)/len(sequence), places=5)
            with tf.GradientTape() as tape:
                variable = tf.Variable(logits)
                loss = train.phone_loss(target, variable)
            self.assertTrue(np.isfinite(tape.gradient(loss, variable).numpy()).all())


if __name__ == '__main__':
    unittest.main()
