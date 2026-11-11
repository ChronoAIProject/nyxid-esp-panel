import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
from batchnorm_policy import apply_policy,snapshot
from phrase_auxiliary import PhraseAuxiliary
for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
tf.keras.utils.set_random_seed(726)
r=json.loads(Path('research/wakeword_p4/results/4090-v30-training.json').read_text());p=argparse.ArgumentParser();mixednet.model_parameters(p)
m=mixednet.model(p.parse_args(r['architecture_args']),(150,40),None);m.load_weights('.local/wakeword/model-gpu-v30-ami-extra/best.weights.h5');apply_policy(m,'frozen');bn=snapshot(m)
m.compile(optimizer=tf.keras.optimizers.Adam(1e-3,global_clipnorm=5),loss='binary_crossentropy')
a=PhraseAuxiliary(m,Path('.local/wakeword/features-gpu-ami-extra-v1'),Path('.local/wakeword/training-phrase-crop-support-v1.json'),.1)
x,labels=a.batch();assert x.shape==(24,150,40);assert np.bincount(labels).tolist()==[6]*4
pred=m(x,training=False).numpy();joint=a.network(x,training=False);np.testing.assert_array_equal(pred,joint[0].numpy());assert joint[1].shape==(24,4)
with tf.GradientTape() as tape:
 _,logits=a.network(x,training=False);loss=a.ce(labels,logits)
g=tape.gradient(loss,m.trainable_variables);nonzero=sum(v is not None and float(tf.reduce_sum(tf.abs(v)))>0 for v in g);assert nonzero>0
before=[v.numpy().copy() for v in a.network.trainable_variables];y=(labels==0).astype(np.float32)[:,None];value=float(a.step(x,y,np.ones(24,dtype=np.float32),x,labels));assert np.isfinite(value);assert bn==snapshot(m)
assert any(not np.array_equal(b,v.numpy()) for b,v in zip(before,a.network.trainable_variables))
assert m.output_shape[-1]==1 and m.count_params()==22801
out={'role':'Actual training feature auxiliary gradient/shape smoke check; not recognition quality','finite_combined_loss':value,'auxiliary_gradient_reaches_binary_model_variables':nonzero,'frozen_bn_unchanged':True,'binary_parameters':m.count_params(),'binary_output_shape':m.output_shape,'class_counts':np.bincount(labels).tolist(),'binary_and_joint_inference_identical_before_step':True,'auxiliary':a.report,'source_sha256':hashlib.sha256(Path('research/wakeword_p4/training/phrase_auxiliary.py').read_bytes()).hexdigest(),'limits':'One batch on existing training features and v30 weights; not independent evaluation or deployed export proof.'}
Path('research/wakeword_p4/results/4090-phrase-auxiliary-smoke-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
