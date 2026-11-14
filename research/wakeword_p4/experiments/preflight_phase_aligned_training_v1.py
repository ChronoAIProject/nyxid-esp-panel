"""Frozen-init gradients and two-update/twofresh RTX4090 phase experiment."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from common_carrier_training_batches_v1 import Batches
from phase_aligned_batches_v1 import phase_input
from hey_batch_schedule_v1 import batch_digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-phase-aligned-preflight-execution-v1.json';OUT=Path('.local/wakeword/phase-aligned-preflight-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def main(action):
 if action=='lock':
  parent=R/'4090-phase-aligned-schedule-execution-v1.json';schedule=R/'4090-phase-aligned-schedule-result-v1.json';s=json.loads(schedule.read_text());assert s['batches']==10000 and s['all1280000_rows_independently_checked'] and s['labels_weights_exact']
  files=json.loads(parent.read_text())['files']
  for p in (parent,schedule,Path(s['schedule_path']),Path(__file__),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py')):files[str(p)]=sha(p)
  for f,h in files.items():assert sha(f)==h,f
  protocol=json.loads((R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json').read_text());initial=next(m for m in protocol['initial_models'] if m['target']=='hey_nyx')
  with LOCK.open('x') as f:json.dump(dict(files=files,initial=initial,optimizer=protocol['optimizer'],schedule=s['schedule_path'],policy='50frozen-init gradient batches/2conditions then2update x2fresh x2conditions. All original15trainables/frozenBN, rawBCE, oneAdamupdate/step; never adjust loss/optimizer/inputs after observation. No formal fit/export/heldout calls.'),f,indent=2)
 else:
  assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
  for f,h in p['files'].items():assert sha(f)==h,f
  OUT.mkdir(exist_ok=False)
  import hey_broad_gpu_training_factory_v1 as helper
  tf=helper.tf;schedule=[json.loads(l) for l in Path(p['schedule']).read_text().splitlines()];initial=p['initial']
  assert sha(initial['initial_weights_path'])==initial['initial_weights_sha256']
  def create():
   m=helper.create(225);m.load_weights(initial['initial_weights_path'])
   for layer in m.layers:
    cell=getattr(layer,'cell',layer)
    if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
   assert len(m.trainable_variables)==15 and all('GPU:0' in v.value.device for v in m.trainable_variables)
   return m,tf.keras.Model(m.inputs,m.layers[-1].input),m.layers[-1]
  def gradients(model,embedding,head,x,y,w):
   with tf.GradientTape() as tape:
    z=tf.linalg.matmul(embedding(tf.constant(x),training=False),head.kernel)+head.bias
    loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
   gs=tape.gradient(loss,model.trainable_variables);assert all(g is not None for g in gs)
   tf.debugging.assert_all_finite(loss,'loss')
   for g in gs:tf.debugging.assert_all_finite(g,'gradient')
   return loss,gs
  model,emb,head=create();sig=signature(model.variables);reader=Batches('hey_nyx','command_context');probe=[]
  for i in range(50):
   (x,y,w),receipt=reader.next();assert receipt==schedule[i]['original_receipt'];batch=[];arrays={}
   for condition in ('nominal','phase'):
    values=x if condition=='nominal' else phase_input(x);assert batch_digest(values,y,w)==schedule[i][condition+'_sha256']
    loss,gs=gradients(model,emb,head,values,y,w);flat=np.concatenate([g.numpy().reshape(-1) for g in gs]);arrays[condition]=flat
    assert signature(model.variables)==sig
    batch.append(dict(condition=condition,loss=float(loss),gradient_l2=float(np.linalg.norm(flat.astype(float))),gradient_sha256=hashlib.sha256(flat.tobytes()).hexdigest()))
   path=OUT/f'gradient-{i+1:02}.npz';np.savez_compressed(path,**arrays)
   probe.append(dict(step=i+1,conditions=batch,gradient_path=str(path),gradient_sha256=sha(path),gradient_cosine=float(np.dot(arrays['nominal'].astype(float),arrays['phase'].astype(float))/(np.linalg.norm(arrays['nominal'].astype(float))*np.linalg.norm(arrays['phase'].astype(float))))))
  reader.log.close();assert signature(model.variables)==sig
  smoke=[]
  for condition in ('nominal','phase'):
   repeats=[]
   for repeat in range(2):
    m,e,h=create();assert signature(m.variables)==sig;frozen=signature(m.non_trainable_variables);optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(m.trainable_variables);reader=Batches('hey_nyx','command_context');history=[]
    for i in range(2):
     (x,y,w),receipt=reader.next();assert receipt==schedule[i]['original_receipt'];values=x if condition=='nominal' else phase_input(x);digest=batch_digest(values,y,w);assert digest==schedule[i][condition+'_sha256']
     loss,gs=gradients(m,e,h,values,y,w);optimizer.apply_gradients(zip(gs,m.trainable_variables));assert int(optimizer.iterations)==i+1 and signature(m.non_trainable_variables)==frozen and all(np.isfinite(v.numpy()).all() for v in m.variables)
     history.append(dict(step=i+1,loss=float(loss),tensor_sha256=digest))
    reader.log.close();assert signature(m.variables)!=sig
    dest=OUT/f'{condition}-{repeat}';dest.mkdir();m.save_weights(dest/'final.weights.h5');np.savez_compressed(dest/'optimizer.npz',**{f'v{j}':v.numpy() for j,v in enumerate(optimizer.variables)})
    repeats.append(dict(initial_signature=sig,final_signature=signature(m.variables),optimizer_signature=signature(optimizer.variables),optimizer_iterations=int(optimizer.iterations),frozen_bn_exact=True,history=history,weights_path=str(dest/'final.weights.h5'),weights_sha256=sha(dest/'final.weights.h5'),optimizer_path=str(dest/'optimizer.npz'),optimizer_sha256=sha(dest/'optimizer.npz')))
   assert all(repeats[0][k]==repeats[1][k] for k in ('initial_signature','final_signature','optimizer_signature','history'))
   smoke.append(dict(condition=condition,repeats=repeats,repeat_exact=True))
  report=dict(execution_sha256=sha(LOCK),gpu=helper.identity,initial_signature=sig,gradient_probe=probe,gradient_variables_unchanged=True,smoke=smoke,all_passed=True,formal_training_started=False,limits=p['policy'])
  with (R/'4090-phase-aligned-preflight-result-v1.json').open('x') as f:json.dump(report,f,indent=2)
  print(json.dumps(dict(all_passed=True,gradient_batches=50,smoke_updates=8,gpu=helper.identity)),flush=True)
if __name__=='__main__':main(sys.argv[1])
