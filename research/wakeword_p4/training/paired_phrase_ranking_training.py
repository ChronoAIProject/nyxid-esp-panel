"""Matched control/treatment: same supervised views and explicit ranking backwards."""
import hashlib
from pathlib import Path
import tensorflow as tf
from paired_rendering_sampler import PairedRenderingSampler
from phrase_ranking import phrase_ranking, verified_phrase_groups

class PairedPhraseRankingTraining:
    def __init__(self, model, coverage, pack, coefficient):
        if coefficient not in (0., 1.):
            raise ValueError('Only fixed control0/treatment1 permitted')
        self.model = model
        self.sampler = PairedRenderingSampler(pack, coverage.raw)
        self.head = model.layers[-1]
        assert isinstance(self.head, tf.keras.layers.Dense) and tuple(self.head.input.shape) == (None, 300)
        self.extractor = tf.keras.Model(model.inputs, self.head.input)
        self.coefficient = float(coefficient)
        self.head_ids = {id(v) for v in self.head.trainable_variables}
        self.encoder_indices = [i for i, v in enumerate(model.trainable_variables) if id(v) not in self.head_ids]
        assert not model.losses
        assert all(not l.trainable for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization))
        model.optimizer.build(model.trainable_variables)
        self.report = dict(objective='phrase-ranking', coefficient=self.coefficient, pair_batch_size=40, positive_windows=10,
            negative_windows=30, phrase_groups=10, seed=227726, pair_bce_coefficient=1.,
            loss='mean128 weighted BCE + mean40 weighted BCE + coefficient*mean10 same-source four-phrase ranking',
            original_head_unchanged=True, single_optimizer_update=True,
            both_arms_explicit_supervised_and_ranking_backwards=True,
            sampler_source_sha256=hashlib.sha256(Path(__file__).with_name('paired_rendering_sampler.py').read_bytes()).hexdigest(),
            loss_source_sha256=hashlib.sha256(Path(__file__).with_name('phrase_ranking.py').read_bytes()).hexdigest())

    def group_indices(self):
        records = [self.sampler.lookup[v, self.sampler.last_trace['speed'], p, a]
            for v in self.sampler.last_trace['voices'] for p in self.sampler.phrases for a in self.sampler.arms]
        return verified_phrase_groups(records)

    @tf.function
    def step(self, x, y, weights, sx, sy, sw, group_indices):
        variables = self.model.trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            main_e = self.extractor(x, training=True)
            pair_e = self.extractor(sx, training=True)
            main_logits = tf.matmul(main_e, self.head.kernel)+self.head.bias
            pair_logits = tf.matmul(pair_e, self.head.kernel)+self.head.bias
            main = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=main_logits)[:, 0]*weights)
            pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sy, logits=pair_logits)[:, 0]*sw)
            supervised = main+pair
            d = phrase_ranking(pair_logits, group_indices)
        gs = tape.gradient(supervised, variables)
        gd = tape.gradient(d, variables)
        assert all(g is not None for g in gs)
        assert all(g is not None for g in gd)
        assert all(gd[i] is not None for i in self.encoder_indices)
        gd = [tf.zeros_like(v) if g is None else g for g, v in zip(gd, variables)]
        for g in gs+gd: tf.debugging.assert_all_finite(g, 'Nonfinite matched rendering gradient')
        # Return both gradient norms to force ranking backward evaluation also in control.
        ns = tf.linalg.global_norm([gs[i] for i in self.encoder_indices])
        nd = tf.linalg.global_norm([gd[i] for i in self.encoder_indices])
        combined = [a+self.coefficient*b for a, b in zip(gs, gd)]
        total = supervised+self.coefficient*d
        tf.debugging.assert_all_finite(total, 'Nonfinite matched rendering loss')
        self.model.optimizer.apply_gradients(zip(combined, variables))
        return total, main, pair, d, ns, nd
