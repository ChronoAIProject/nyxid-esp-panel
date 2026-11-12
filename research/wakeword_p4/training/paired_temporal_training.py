"""Matched temporal-risk arms: original main, last-view BCE/ranking, one update."""
import hashlib
from pathlib import Path
import tensorflow as tf
from paired_phrase_ranking_training import PairedPhraseRankingTraining
from paired_temporal_sampler import PairedTemporalSampler
from phrase_ranking import phrase_ranking,verified_phrase_groups
from temporal_tail_risk import temporal_tail_risk


class PairedTemporalTraining(PairedPhraseRankingTraining):
    def __init__(self,model,coverage,pack,coefficient):
        super().__init__(model,coverage,pack,coefficient)
        self.sampler=PairedTemporalSampler(pack,coverage.raw)
        self.report.update(objective='temporal-tail-risk',ranking_coefficient=1.,temporal_coefficient=coefficient,
            temporal_windows=5,seed=237726,
            loss='main128 weighted BCE + last40 weighted BCE + last10 phrase ranking + coefficient*mean40 weighted logmeanexp5 frame BCE',
            sampler_source_sha256=hashlib.sha256(Path(__file__).with_name('paired_temporal_sampler.py').read_bytes()).hexdigest(),
            temporal_loss_source_sha256=hashlib.sha256(Path(__file__).with_name('temporal_tail_risk.py').read_bytes()).hexdigest(),
            both_arms_explicit_base_ranking_temporal_backwards=True)

    def group_indices(self):
        return verified_phrase_groups(self.sampler.last_records)

    @tf.function
    def step(self,x,y,weights,sx,sy,sw,group_indices):
        variables=self.model.trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            main_e=self.extractor(x,training=True)
            pair_e=self.extractor(tf.reshape(sx,[200,150,40]),training=True)
            main_logits=tf.matmul(main_e,self.head.kernel)+self.head.bias
            logits=tf.reshape(tf.matmul(pair_e,self.head.kernel)+self.head.bias,[40,5])
            main=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=main_logits)[:,0]*weights)
            pair=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sy,logits=logits[:,-1])*sw)
            ranking=phrase_ranking(logits[:,-1],group_indices)
            base=main+pair+ranking
            risk=temporal_tail_risk(logits,sy,sw)
        gb=tape.gradient(base,variables);gr=tape.gradient(ranking,variables);gt=tape.gradient(risk,variables)
        assert all(g is not None for g in gb+gr+gt)
        for g in gb+gr+gt:tf.debugging.assert_all_finite(g,'Nonfinite temporal matched gradient')
        def norm(gradients):return tf.linalg.global_norm([gradients[i] for i in self.encoder_indices])
        nb,nr,nt=norm(gb),norm(gr),norm(gt)
        total=base+self.coefficient*risk
        tf.debugging.assert_all_finite(total,'Nonfinite temporal matched loss')
        self.model.optimizer.apply_gradients(zip([a+self.coefficient*b for a,b in zip(gb,gt)],variables))
        return total,main,pair,ranking,nb,nr,risk,nt
