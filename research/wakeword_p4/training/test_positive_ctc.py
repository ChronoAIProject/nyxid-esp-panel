"""Crop boundaries, independent keyword-path math and exact zero-gain replay."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from background_ctc import mark_ambient
from positive_ctc import eligible, mark, mask, keyword_loss, Exposure
from test_background_ctc import fixture
from phrase_ctc_readout import targets


def row(**changes):
    return dict(split='training', label=1, lead_frame=400, keyword_frame=451,
                engine='piper', text='Hey Nyx, check the volume.', sha256='a'*64,
                variant=0, **changes)


class PositiveTest(unittest.TestCase):
    def test_crop_boundaries_without_resampling(self):
        source = row()
        self.assertFalse(eligible(source, 453, 335))
        self.assertTrue(eligible(source, 454, 335))
        self.assertTrue(eligible(source, 735, 335))
        self.assertFalse(eligible(source, 736, 335))
        for change in ({'split': 'validation'}, {'label': 0}):
            invalid = {**source, **change}
            with self.assertRaises(ValueError):
                eligible(invalid, 454, 335)

    def test_marker_shuffle_and_actual_exposure_trace(self):
        labels = np.zeros((128, 2), np.float32)
        labels[:32, 0] = 1
        labels[:8, 1] = np.arange(1, 17, 2)
        labels[32:40, 1] = np.arange(2, 17, 2)
        sources = [row() for _ in range(32)]
        selections = [(r, 453 if i % 2 else 454) for i, r in enumerate(sources)]
        marked = mark(mark_ambient(labels, 48), selections, 335)
        expected = np.zeros(128, bool)
        expected[8:32:2] = True
        order = np.random.default_rng(4).permutation(128)
        np.testing.assert_array_equal(mask(marked[order]), expected[order])
        np.testing.assert_array_equal(marked[:, :2], labels)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'trace.jsonl'
            exposure = Exposure(sources, path)
            samples = np.arange(128*335*40, dtype=np.uint16).reshape(128, 335, 40)
            exposure.record(1, marked, selections, samples)
            meta = exposure.metadata()
            exposure.stream.close()
            trace = json.loads(path.read_bytes())
            self.assertEqual(meta['eligible_examples'], 12)
            self.assertEqual(meta['examples'], 24)
            self.assertEqual(meta['eligible_distinct_sources'], 1)
            self.assertEqual(meta['trace_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual([d[:3] for d in trace['draws']],
                             [[i, selections[i][1], int(expected[i])] for i in range(8, 32)])
            self.assertEqual(trace['draws'][0][3], hashlib.sha256(samples[8].tobytes()).hexdigest())

    def test_rejects_negative_paired_and_ambient_markers(self):
        for labels in ([[0, 0, 0, 1]], [[1, 2, 0, 1]], [[1, 0, 1, 1]],
                       [[1, 0, 0, .5]], [[1, 0, 0, np.nan]], [[1, 0, 0]]):
            with self.subTest(labels=labels), self.assertRaises(tf.errors.InvalidArgumentError):
                mask(tf.constant(labels, tf.float32))
        for labels in ([[0, 0, 0]], [[1, 0, 1]], [[1, 0]]):
            with self.assertRaises(ValueError):
                mark(np.array(labels, np.float32), [(row(), 454)], 335)

    def test_keyword_loss_matches_exhaustive_paths_and_handles_empty(self):
        logits = np.random.default_rng(7).normal(size=(2, 4, 3)).astype(np.float32)
        probabilities = np.exp(logits.astype(np.float64)-logits.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        totals = []
        for sample in probabilities:
            total = 0.
            for path in itertools.product(range(3), repeat=4):
                collapsed = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]
                if collapsed == [1, 2]:
                    total += np.prod([sample[i, v] for i, v in enumerate(path)])
            totals.append(-np.log(total)/2)
        self.assertAlmostEqual(float(keyword_loss(tf.constant(logits))), np.mean(totals), places=6)
        self.assertEqual(float(keyword_loss(tf.zeros((0, 67, 6)))), 0.)

    def test_combined_loss_and_empty_cohort(self):
        for include in (False, True):
            model, train = fixture(1., 1.)
            x = np.random.default_rng(31).normal(size=(8, 335, 40)).astype(np.float32)
            labels = np.array([[1, 1], [0, 2], [0, 3], [0, 4], [1, 0],
                               [0, 0], [0, 0], [0, 0]], np.float32)
            marked = np.column_stack((mark_ambient(labels, 2), np.zeros(8, np.float32)))
            marked[4, 3] = include
            weights = np.where(labels[:, 0] == 1, 1, 5).astype(np.float32)
            probability, logits = train.auxiliary_model(x)
            binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels[:, :1], probability)*weights)
            pair = tf.reduce_mean(tf.nn.ctc_loss(targets(np.arange(4)), logits[:4], [2]*4,
                [67]*4, logits_time_major=False, blank_index=0))/2
            blank = -tf.reduce_mean(tf.math.log(tf.nn.softmax(logits[-2:], axis=-1)[..., 0]))
            positive = tf.nn.ctc_loss([[1, 2]], logits[4:5], [2], [67],
                logits_time_major=False, blank_index=0)[0]/2 if include else 0.
            self.assertAlmostEqual(float(train(x, marked, weights)), float(binary+.1*(pair+blank+positive)), places=5)
            self.assertEqual(train.positive_metadata()['examples'], int(include))
            self.assertTrue(all(np.isfinite(v.numpy()).all() for v in model.weights))

    def test_zero_exactly_replays_background_parent_weights_and_adam(self):
        old, old_train = fixture(1.)
        new, new_train = fixture(1., 0.)
        labels = mark_ambient(np.array([[1, 1], [0, 2], [0, 3], [0, 4],
                                        [1, 0], [0, 0], [0, 0]], np.float32), 2)
        marked = np.column_stack((labels, [0, 0, 0, 0, 1, 0, 0])).astype(np.float32)
        weights = np.where(labels[:, 0] == 1, 1, 5).astype(np.float32)
        rng = np.random.default_rng(80)
        for _ in range(4):
            x = rng.normal(size=(7, 335, 40)).astype(np.float32)
            self.assertEqual(float(old_train(x, labels, weights)), float(new_train(x, marked, weights)))
            for a, b in zip(old_train.auxiliary_model.weights, new_train.auxiliary_model.weights, strict=True):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            for a, b in zip(old.optimizer.variables, new.optimizer.variables, strict=True):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertEqual(new_train.positive_metadata()['examples'], 4)

    def test_invalid_cli_before_input_output_access(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            common = [sys.executable, str(Path(__file__).with_name('train_stream.py')),
                '--upstream', str(root/'missing'), '--features', str(root/'missing'),
                '--output', str(root/'output')]
            valid = ['--compact-phrase-plan', str(root/'missing'), '--phone-background-gain', '1', '--phone-gain', '.1']
            for extra in (['--phone-positive-gain', '1'],
                          valid+['--phone-positive-gain', 'nan'], valid+['--phone-positive-gain', '-.1'],
                          valid+['--phone-positive-gain', '1.1'],
                          valid+['--phone-gain', '0', '--phone-positive-gain', '1']):
                run = subprocess.run(common+extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn('Positive CTC requires', run.stderr)
                self.assertNotIn('Traceback', run.stderr)
                self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
