"""v39 training-only sequence supplement with one shared optimizer update."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sequence_loss import build_sequence_logits, mean_probability_bce

class FiveScoreTraining:
    def __init__(self, model, root):
        root=Path(root)
        digest=hashlib.sha256((root/'training.npz').read_bytes()).hexdigest()
        expected='e5e731ff9fe390100b58d153002648cc238edf6e7daefcc58ea39929d256efc7'
        report=json.loads((root/'report.json').read_text())
        if digest!=expected or report['pack_sha256']!=expected or report['final_data_used'] or report['wake_models_scored_for_inclusion']:
            raise ValueError('Require pinned training-only sequence pack')
        z=np.load(root/'training.npz',allow_pickle=False);self.x=z['x'];self.y=z['y'];self.groups={}
        if self.x.shape!=(817,162,40) or self.x.dtype!=np.uint16:raise ValueError('Sequence pack contract')
        for row in report['records']:
            i=row['sequence_index']
            if i is None:continue
            if not row['all_five_whole_signal_valid'] or self.y[i]!=int(row['target']=='hey_nyx'):raise ValueError('Sequence scope/label')
            self.groups.setdefault(row['target'],{}).setdefault(row['source_group'],{}).setdefault(row['voice'],[]).append(i)
        if set(self.groups)!={'hey_nyx','hey_nick','hey_next','hey_mix'} or any(set(g)!={'kokoro','alba','cori'} for g in self.groups.values()):
            raise ValueError('Missing phrase/checkpoint group')
        self.rng=np.random.default_rng(117726);self.model=model
        self.logits=build_sequence_logits(model)
        model.optimizer.build(model.trainable_variables)
        if model.losses:raise ValueError('No additional regularization losses permitted')
        self.report={'pack_sha256':digest,'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
            'seed':117726,'batch_size':16,'positive_slots':8,'negative_slots':8,'negative_weight':20,
            'loss_weight':1.,'loss':'main128 weighted BCE + sequence16 weighted stable BCE(mean five sigmoid probabilities)',
            'single_optimizer_update':True,'groups':report['groups']}

    def batch(self):
        ids=[]
        for i in range(16):
            target='hey_nyx' if i<8 else str(self.rng.choice(['hey_nick','hey_next','hey_mix']))
            source=str(self.rng.choice(['kokoro','alba','cori']))
            voices=self.groups[target][source];voice=str(self.rng.choice(sorted(voices)))
            ids.append(int(self.rng.choice(voices[voice])))
        y=self.y[ids].astype(np.float32)
        return np.clip(self.x[ids].astype(np.float32)/25.6,0,26),y,np.where(y>0,1.,20.).astype(np.float32)

    @tf.function
    def step(self,x,y,weights,sx,sy,sw):
        with tf.GradientTape() as tape:
            predictions=self.model(x,training=True)
            main=tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,predictions)*weights)
            logits=self.logits(sx,training=True)
            tf.debugging.assert_equal(tf.shape(logits)[1],5)
            extra=tf.reduce_mean(mean_probability_bce(sy,logits)*sw)
            loss=main+extra
        gradients=tape.gradient(loss,self.model.trainable_variables)
        for g in gradients:
            if g is None:raise ValueError('Disconnected original variable')
            tf.debugging.assert_all_finite(g,'Nonfinite sequence update')
        tf.debugging.assert_all_finite(loss,'Nonfinite loss')
        self.model.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        return loss
