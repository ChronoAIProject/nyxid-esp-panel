"""Bounded GPU update feasibility for frozen pretrained encoder vs full tuning."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
R=Path('research/wakeword_p4/results')
LOCK=R/'4090-frozen-encoder-smoke-execution-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sig(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def main():
 if sys.argv[1]=='lock':
  init=R/'4090-hey-broad-context225-initialization-v1.json';a=json.loads(init.read_text())
  files=[Path(__file__),init,Path(a['snapshot']['path'])]+[Path(__file__).with_name(n) for n in ['hey_broad_gpu_training_factory_v1.py','common_carrier_training_batches_v1.py','phase_aligned_batches_v1.py']]
  d={'files':{str(p):sha(p) for p in files},'initial':a['snapshot'],'targets':['hey_nyx','hey_willow'],'conditions':['full','frozen'],'repeats':2,'steps':2,'policy':'Two updates per fresh arm only, shared original context225 pretrained encoder/seed726 head snapshot. Original command_context sampler, phase224+zero all slots, raw BCE positive1/negative20, Adam1e-5/.9/.999/eps1e-7. Full15 trainables/frozenBN versus only2 Dense variables/all encoder frozen. No formal fit, export, thresholds, new data or efficacy claim.'}
  assert sha(a['snapshot']['path'])==a['snapshot']['sha256']
  with LOCK.open('x') as f:json.dump(d,f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 import hey_broad_gpu_training_factory_v1 as h
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 tf=h.tf;out=Path('.local/wakeword/frozen-encoder-smoke-v1');out.mkdir(exist_ok=False);results=[]
 for target in d['targets']:
  for condition in d['conditions']:
   repeats=[]
   for repeat in range(2):
    m=h.create(225);m.load_weights(d['initial']['path']);head=m.layers[-1]
    for l in m.layers:
     c=getattr(l,'cell',l)
     if isinstance(c,tf.keras.layers.BatchNormalization) or (condition=='frozen' and l is not head):l.trainable=c.trainable=False
    vs=m.trainable_variables;assert len(vs)==(2 if condition=='frozen' else 15)
    encoder=[v for v in m.variables if all(v is not w for w in head.variables)]
    before=sig(encoder);bn=sig(m.non_trainable_variables);initial=sig(m.variables)
    emb=tf.keras.Model(m.inputs,head.input);opt=tf.keras.optimizers.Adam(1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7);opt.build(vs)
    reader=Batches(target,'command_context');history=[]
    for step in range(1,3):
     (x,y,w),receipt=reader.next();x=phase_input(x)
     with tf.GradientTape() as tape:
      z=tf.linalg.matmul(emb(tf.constant(x),training=False),head.kernel)+head.bias
      loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
     gs=tape.gradient(loss,vs);assert all(g is not None for g in gs)
     for g in gs:tf.debugging.assert_all_finite(g,'gradient')
     tf.debugging.assert_all_finite(loss,'loss');opt.apply_gradients(zip(gs,vs));assert int(opt.iterations)==step
     assert sig(m.non_trainable_variables)==bn
     if condition=='frozen':assert sig(encoder)==before
     history.append({'step':step,'loss':float(loss),'original_batch_sha256':receipt['tensor_sha256'],'phase_input_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    reader.log.close()
    dest=out/f'{target}-{condition}-{repeat}.weights.h5';m.save_weights(dest)
    repeats.append({'initial_signature':initial,'final_signature':sig(m.variables),'optimizer_signature':sig(opt.variables),'encoder_before':before,'encoder_after':sig(encoder),'history':history,'weights_path':str(dest),'weights_sha256':sha(dest),'trainables':len(vs)})
   assert all(repeats[0][k]==repeats[1][k] for k in ['initial_signature','final_signature','optimizer_signature','history','encoder_before','encoder_after'])
   results.append({'target':target,'condition':condition,'repeats':repeats,'repeat_exact':True});print(target,condition,'passed',flush=True)
 for target in d['targets']:
  arms=[a['repeats'][0] for a in results if a['target']==target]
  assert arms[0]['initial_signature']==arms[1]['initial_signature']
  assert [r['phase_input_sha256'] for r in arms[0]['history']]==[r['phase_input_sha256'] for r in arms[1]['history']]
 result={'execution_sha256':sha(LOCK),'cases':results,'gpu':h.identity,'all_passed':True,'limits':d['policy']}
 with (R/'4090-frozen-encoder-smoke-result-v1.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
