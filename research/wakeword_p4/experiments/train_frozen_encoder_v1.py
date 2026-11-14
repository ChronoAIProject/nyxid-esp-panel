"""Fixed-final four-arm encoder-freezing contrast. Never resumes or restarts."""
import hashlib,importlib.metadata,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-frozen-encoder-formal-execution-v1.json';OUT=Path('.local/wakeword/frozen-encoder-formal-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sig(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def main():
 if sys.argv[1]=='lock':
  parent=R/'4090-frozen-encoder-schedule-execution-v1.json';d=json.loads(parent.read_text());files=d['files']
  paths=[parent,R/'4090-frozen-encoder-schedule-result-v1.json',R/'4090-frozen-encoder-smoke-execution-v1.json',R/'4090-frozen-encoder-smoke-result-v1.json',R/'4090-frozen-encoder-smoke-independent-audit-v1.json',Path(__file__),Path(__file__).with_name('frozen_encoder_formal_v1.md'),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),Path(__file__).with_name('audit_frozen_encoder_formal_v1.py')]
  schedules=json.loads(paths[1].read_text());smoke=json.loads(paths[3].read_text());audit=json.loads(paths[4].read_text());assert schedules['all_passed'] and smoke['all_passed'] and audit['all_passed'] and audit['source_result_sha256']==sha(paths[3])
  old=json.loads(paths[2].read_text());initial=old['initial'];paths.append(Path(initial['path']))
  for p in paths:files[str(p)]=sha(p)
  for s in schedules['schedules'].values():assert sha(s['path'])==s['sha256'];files[s['path']]=s['sha256']
  for p,h in files.items():assert sha(p)==h,p
  d.update(initial=initial,schedules=schedules['schedules'],steps=10000,targets=['hey_nyx','hey_willow'],conditions=['full','frozen'],repeats=1,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']},smoke_result=str(paths[3]))
  with LOCK.open('x') as f:json.dump(d,f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 import hey_broad_gpu_training_factory_v1 as h
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 from hey_batch_schedule_v1 import batch_digest
 tf=h.tf;smoke=json.loads(Path(d['smoke_result']).read_text());OUT.mkdir(exist_ok=False);cases=[];current=None
 try:
  for target in d['targets']:
   schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];assert len(schedule)==10000
   for condition in d['conditions']:
    dest=OUT/f'{target}-{condition}';dest.mkdir();m=h.create(225);m.load_weights(d['initial']['path']);head=m.layers[-1]
    for l in m.layers:
     c=getattr(l,'cell',l)
     if isinstance(c,tf.keras.layers.BatchNormalization) or (condition=='frozen' and l is not head):l.trainable=c.trainable=False
    vs=m.trainable_variables;assert len(vs)==(2 if condition=='frozen' else 15)
    expected=next(a for a in smoke['cases'] if a['target']==target and a['condition']==condition)['repeats'][0]
    assert sig(m.variables)==expected['initial_signature'];m.save_weights(dest/'initial.weights.h5')
    frozen=sig(m.non_trainable_variables);emb=tf.keras.Model(m.inputs,head.input);opt=tf.keras.optimizers.Adam(1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7);opt.build(vs)
    reader=Batches(target,'command_context');started=time.monotonic();current={'target':target,'condition':condition,'step':0}
    with (dest/'history.jsonl').open('x') as log:
     for step in range(1,10001):
      (x,y,w),receipt=reader.next();x=phase_input(x);want=schedule[step-1];assert receipt==want['original_receipt'] and batch_digest(x,y,w)==want['phase_sha256']
      with tf.GradientTape() as tape:
       z=tf.linalg.matmul(emb(tf.constant(x),training=False),head.kernel)+head.bias
       loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
      gs=tape.gradient(loss,vs);assert all(g is not None for g in gs)
      for g in gs:tf.debugging.assert_all_finite(g,'gradient')
      tf.debugging.assert_all_finite(loss,'loss');opt.apply_gradients(zip(gs,vs));assert int(opt.iterations)==step and sig(m.non_trainable_variables)==frozen
      assert all(np.isfinite(v.numpy()).all() for v in m.variables)
      row={'step':step,'loss':float(loss),'original_batch_sha256':receipt['tensor_sha256'],'phase_input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'tensor_sha256':want['phase_sha256']};log.write(json.dumps(row)+'\n');log.flush();current['step']=step
      if step<=2:assert {k:row[k] for k in expected['history'][step-1]}==expected['history'][step-1]
      if step==2:assert sig(m.variables)==expected['final_signature'] and sig(opt.variables)==expected['optimizer_signature']
      if step%1000==0:m.save_weights(dest/f'step{step}.weights.h5');np.savez_compressed(dest/f'step{step}-optimizer.npz',**{f'v{i}':v.numpy() for i,v in enumerate(opt.variables)})
      if step<=2 or step%100==0:print(json.dumps(dict(target=target,condition=condition,**row)),flush=True)
    reader.finish();m.save_weights(dest/'final.weights.h5');np.savez_compressed(dest/'final-optimizer.npz',**{f'v{i}':v.numpy() for i,v in enumerate(opt.variables)})
    result={'target':target,'condition':condition,'steps':10000,'weights_path':str(dest/'final.weights.h5'),'weights_sha256':sha(dest/'final.weights.h5'),'optimizer_path':str(dest/'final-optimizer.npz'),'optimizer_sha256':sha(dest/'final-optimizer.npz'),'history_path':str(dest/'history.jsonl'),'history_sha256':sha(dest/'history.jsonl'),'elapsed_seconds':time.monotonic()-started,'final_signature':sig(m.variables),'optimizer_signature':sig(opt.variables),'frozen_signature':frozen,'optimizer_iterations':int(opt.iterations),'trainables':len(vs)}
    (dest/'training.json').write_text(json.dumps(result,indent=2)+'\n');cases.append(result)
  with (R/'4090-frozen-encoder-formal-result-v1.json').open('x') as f:json.dump({'execution_sha256':sha(LOCK),'cases':cases,'gpu':h.identity,'all_passed':True,'limits':'Training completed only; no numeric, recognition, INT8 or acceptance qualification.'},f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());(OUT/'failure.json').write_text(json.dumps({'current':current,'completed':cases,'automatic_restart':False},indent=2));raise
if __name__=='__main__':main()
