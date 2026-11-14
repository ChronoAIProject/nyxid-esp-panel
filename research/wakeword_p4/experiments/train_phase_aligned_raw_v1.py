"""Fixed-final matched nominal/phase Nyx raw training; never auto-restarts."""
import hashlib,importlib.metadata,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from common_carrier_training_batches_v1 import Batches
from phase_aligned_batches_v1 import phase_input
from hey_batch_schedule_v1 import batch_digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-phase-aligned-formal-execution-v1.json';OUT=Path('.local/wakeword/phase-aligned-formal-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def main(action):
 if action=='lock':
  parent=R/'4090-phase-aligned-preflight-execution-v1.json';result=R/'4090-phase-aligned-preflight-result-v1.json';p=json.loads(parent.read_text());test=json.loads(result.read_text())
  auditpath=R/'4090-phase-aligned-preflight-independent-audit-v1.json';audit=json.loads(auditpath.read_text());assert audit['all_passed'] and audit['original_result_sha256']==sha(result)
  assert test['all_passed'] and test['gradient_variables_unchanged'] and test['execution_sha256']==sha(parent) and len(test['gradient_probe'])==50
  assert len(test['smoke'])==2 and all(a['repeat_exact'] for a in test['smoke'])
  for entry in test['gradient_probe']:assert sha(entry['gradient_path'])==entry['gradient_sha256']
  for a in test['smoke']:
   for r in a['repeats']:assert sha(r['weights_path'])==r['weights_sha256'] and sha(r['optimizer_path'])==r['optimizer_sha256']
  for f in (parent,result,auditpath,Path(__file__)):p['files'][str(f)]=sha(f)
  p.update(versions={n:importlib.metadata.version(n) for n in ('tensorflow','keras','numpy','h5py')},steps=10000,repeats=1,arms=['nominal','phase'],preflight_result=str(result),policy='Fresh nominal/phase Nyx from same original raw init, exact10000 shared command_context inputs/labels/weights/endpoints, freshAdam/rawBCE/frozenBN. Eager training expression as smoke; first2 loss/tensorSHA/final optimizer/model signature must match preflight at step2. Fixedfinal10000 only; snapshots evidence, no score selection/restarts/heldout/export.')
  for f,h in p['files'].items():assert sha(f)==h,f
  with LOCK.open('x') as f:json.dump(p,f,indent=2)
 else:
  assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
  for f,h in p['files'].items():assert sha(f)==h,f
  for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
  OUT.mkdir(exist_ok=False)
  import hey_broad_gpu_training_factory_v1 as helper
  tf=helper.tf;test=json.loads(Path(p['preflight_result']).read_text());schedule=[json.loads(l) for l in Path(p['schedule']).read_text().splitlines()];cases=[];current=None
  try:
   for condition in p['arms']:
    dest=OUT/condition;dest.mkdir();m=helper.create(225);m.load_weights(p['initial']['initial_weights_path'])
    for layer in m.layers:
     cell=getattr(layer,'cell',layer)
     if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
    assert len(m.trainable_variables)==15 and signature(m.variables)==test['initial_signature']
    m.save_weights(dest/'initial.weights.h5');frozen=signature(m.non_trainable_variables);emb=tf.keras.Model(m.inputs,m.layers[-1].input);head=m.layers[-1];optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(m.trainable_variables)
    reader=Batches('hey_nyx','command_context');expected=next(a for a in test['smoke'] if a['condition']==condition)['repeats'][0];started=time.monotonic();current=dict(condition=condition,step=0)
    with (dest/'history.jsonl').open('x') as log:
     for index in range(1,10001):
      (x,y,w),receipt=reader.next();want=schedule[index-1];assert receipt==want['original_receipt'];values=x if condition=='nominal' else phase_input(x);digest=batch_digest(values,y,w);assert digest==want[condition+'_sha256']
      with tf.GradientTape() as tape:
       z=tf.linalg.matmul(emb(tf.constant(values),training=False),head.kernel)+head.bias
       loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
      gs=tape.gradient(loss,m.trainable_variables);assert all(g is not None for g in gs);tf.debugging.assert_all_finite(loss,'loss')
      for g in gs:tf.debugging.assert_all_finite(g,'gradient')
      optimizer.apply_gradients(zip(gs,m.trainable_variables));assert int(optimizer.iterations)==index and signature(m.non_trainable_variables)==frozen and all(np.isfinite(v.numpy()).all() for v in m.variables)
      row=dict(step=index,loss=float(loss),tensor_sha256=digest);log.write(json.dumps(row)+'\n');log.flush();current['step']=index
      if index<=2:assert row==expected['history'][index-1]
      if index==2:assert signature(m.variables)==expected['final_signature'] and signature(optimizer.variables)==expected['optimizer_signature']
      if index%1000==0:m.save_weights(dest/f'step{index}.weights.h5');tf.train.Checkpoint(model=m,optimizer=optimizer).write(str(dest/f'step{index}'))
      if index<=2 or index%100==0:print(json.dumps(dict(condition=condition,**row)),flush=True)
    reader.finish();m.save_weights(dest/'final.weights.h5');np.savez_compressed(dest/'final-optimizer.npz',**{f'v{j}':v.numpy() for j,v in enumerate(optimizer.variables)})
    receipt=dict(condition=condition,steps=10000,initial_signature=test['initial_signature'],final_signature=signature(m.variables),optimizer_signature=signature(optimizer.variables),frozen_bn_exact=True,weights_path=str(dest/'final.weights.h5'),weights_sha256=sha(dest/'final.weights.h5'),history_path=str(dest/'history.jsonl'),history_sha256=sha(dest/'history.jsonl'),optimizer_path=str(dest/'final-optimizer.npz'),optimizer_sha256=sha(dest/'final-optimizer.npz'),elapsed_seconds=time.monotonic()-started)
    (dest/'training.json').write_text(json.dumps(receipt,indent=2)+'\n');cases.append(receipt)
   with (R/'4090-phase-aligned-formal-result-v1.json').open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),cases=cases,all_passed=True,gpu=helper.identity,limits=p['policy']),f,indent=2)
  except Exception:
   (OUT/'failure.log').write_text(traceback.format_exc());(OUT/'failure.json').write_text(json.dumps(dict(current=current,completed=cases,automatic_restart=False),indent=2));raise
if __name__=='__main__':main(sys.argv[1])
