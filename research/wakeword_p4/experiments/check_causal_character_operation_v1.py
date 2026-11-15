"""One GPU training-input forward operation, not an accuracy/numerical gate."""
import json
from pathlib import Path
import numpy as np
from causal_character_supervision_v1 import Sampler
from causal_character_graph_v1 import joint_loss,logits

def main():
    import hey_broad_gpu_training_factory_v1 as factory
    tf=factory.tf;tf.keras.utils.set_random_seed(726)
    plan=json.loads(Path('research/wakeword_p4/results/4090-causal-source-balanced-plan-v1.json').read_text())
    model=factory.create(309);model.load_weights(plan['initial']['snapshot']['path'])
    for layer in model.layers:
        cell=getattr(layer,'cell',layer)
        if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
    with tf.device('/GPU:0'):
        head=tf.keras.layers.Dense(28,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1726),bias_initializer='zeros',name='training_character_head');head.build((None,60))
        (x,p,n),_,rows,t=Sampler().sequence_batch()
        binary,characters=logits(model,head,tf.constant(x),tf)
        loss,bl,cl=joint_loss(model,head,tf.constant(x),tf.constant(p),tf.constant(n),*[tf.constant(t[k]) for k in ['batch_indices','labels','label_lengths','logit_lengths']],tf)
    assert tuple(binary.shape)==(128,282,1) and tuple(characters.shape)==(128,282,28)
    assert head.count_params()==1708 and 'GPU:0' in loss.device and all(np.isfinite(float(v.numpy())) for v in [loss,bl,cl])
    out=dict(schema='causal-character-operation-v1',device=factory.identity,loss_device=loss.device,binary_shape=list(binary.shape),character_shape=list(characters.shape),head_parameters=head.count_params(),original_model_parameters=model.count_params(),known_transcripts=80,unknown_ambient=48,ctc_blank_id=0,total_loss=float(loss.numpy()),binary_loss=float(bl.numpy()),length_normalized_ctc_loss=float(cl.numpy()),optimizer_updates=0,recognition_scored=False,numerical_comparison_performed=False,limits='One actual existing training batch forward/loss interface on RTX4090 only. No optimizer/gradients/CPU-GPU/FP32-INT8 comparison, source substitution, coefficient tuning, TTS or final access. Auxiliary character head not deployed; recognition benefit unproven.')
    path=Path('research/wakeword_p4/results/4090-causal-character-operation-v1.json')
    with path.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
