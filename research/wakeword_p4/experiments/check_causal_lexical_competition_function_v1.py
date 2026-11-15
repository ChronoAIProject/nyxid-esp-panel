"""One actual model/batch functional call, no numerical comparison or gradients."""
import json
from pathlib import Path
import numpy as np
import hey_broad_gpu_training_factory_v1 as factory
from causal_lexical_competition_v1 import create,logits,presentation_loss
from causal_lexical_competition_batch_v1 import Sampler

def main():
    tf=factory.tf;tf.keras.utils.set_random_seed(726)
    parent=json.loads(Path('research/wakeword_p4/results/4090-causal-source-balanced-plan-v1.json').read_text())
    base=factory.create(309);base.load_weights(parent['initial']['snapshot']['path'])
    for layer in base.layers:
        cell=getattr(layer,'cell',layer)
        if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
    model=create(base,tf);(x,p,n,labels),desc=Sampler(726).batch()
    with tf.device('/GPU:0'):
        z=logits(model,tf.constant(x),tf);loss=presentation_loss(z,tf.constant(p),tf.constant(n),tf.constant(labels),tf)
        scalar=model(tf.constant(x[:1,:309]),training=False)
    assert tuple(z.shape)==(128,282,6) and tuple(scalar.shape)==(1,1)
    assert np.isfinite(float(loss.numpy())) and np.isfinite(scalar.numpy()).all()
    out=dict(schema='causal-lexical-competition-function-v1',device=z.device,parameters=model.count_params(),variables=len(model.variables),trainable_variables=len(model.trainable_variables),frozen_variables=len(model.non_trainable_variables),batch_shape=list(x.shape),categorical_logits_shape=list(z.shape),scalar_output_shape=list(scalar.shape),weighted_class_steps={str(c):int(((labels==c)&((p+n)>0)).sum()) for c in range(6)},first_batch_loss=float(loss.numpy()),optimizer_updates=0,comparison_performed=False,recognition_scored=False,policy='One actual training-only batch and scalar interface functional call. No gradient/numerical consistency comparison, export, inference benchmark, training or development/final scoring. This is not acceptance.')
    with Path('research/wakeword_p4/results/4090-causal-lexical-competition-function-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
