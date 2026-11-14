"""Four matched arms, two fresh repeats, two updates; never resumes/retries."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-adversarial-smoke-execution-v1.json';RESULT=R/'4090-engine-adversarial-smoke-result-v1.json';OUT=Path('.local/wakeword/engine-adversarial-smoke-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory(vs):return [dict(index=i,name=str(getattr(v,'path',v.name)),shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for i,v in enumerate(vs)]
def save(p,vs):np.savez_compressed(p,**{f'v{i}':v.numpy() for i,v in enumerate(vs)})
def main():
 if sys.argv[1]=='lock':
  parent=R/'4090-engine-adversarial-gradient-execution-v1.json';d=json.loads(parent.read_text());files=dict(d['files']);probe=R/'4090-engine-adversarial-gradient-result-v1.json';audit=R/'4090-engine-adversarial-gradient-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_passed'] and a['original_result_sha256']==sha(probe)
  old=R/'4090-frozen-encoder-smoke-result-v1.json'
  for p in [parent,probe,audit,old,Path(__file__),Path(__file__).with_name('engine_adversarial_training_core_v1.py'),Path(__file__).with_name('engine_adversarial_smoke_v1.md'),Path(__file__).with_name('audit_engine_adversarial_smoke_v1.py')]:files[str(p)]=sha(p)
  for p,h in files.items():assert sha(p)==h,p
  d.update(files=files,probe_result=str(probe),old_smoke=str(old),coefficient={'control':0.,'adversarial':1.},steps=2,repeats=2,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']})
  with LOCK.open('x') as f:json.dump(d,f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 import engine_adversarial_training_core_v1 as core
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 from hey_batch_schedule_v1 import batch_digest
 tf=core.tf;OUT.mkdir(exist_ok=False);reports=[];probe=json.loads(Path(d['probe_result']).read_text());aux=json.loads(Path(d['input_result']).read_text())['batches'];old=json.loads(Path(d['old_smoke']).read_text());current=None
 try:
  for target in ['hey_nyx','hey_willow']:
   schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];original=next(t for t in probe['targets'] if t['target']==target);first=original['cases'][0];baseline=next(c for c in old['cases'] if c['target']==target and c['condition']=='full')['repeats'][0]
   for condition,coefficient in d['coefficient'].items():
    repetitions=[]
    for repeat in range(2):
     current=dict(target=target,condition=condition,repeat=repeat,step=0);folder=OUT/f'{target}-{condition}-{repeat}';folder.mkdir();b=core.make(d['initial']['path'],d['domain_seed']);m=b['model'];vs=b['variables'];state=b['state'];assert core.sig(m.variables)==baseline['initial_signature'];bn=core.sig(b['bn']);initial=inventory(state);m.save_weights(folder/'initial.weights.h5');save(folder/'initial-state.npz',state);opt=tf.keras.optimizers.Adam(1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7);opt.build(vs);assert len(opt.variables)==36;reader=Batches(target,'command_context');history=[]
     for step in range(2):
      current['step']=step+1;(x,y,w),receipt=reader.next();x=phase_input(x);want=schedule[step];assert receipt==want['original_receipt'] and batch_digest(x,y,w)==want['phase_sha256']
      with np.load(aux[step]['arrays_path'],allow_pickle=False) as a:ax=a['x'].copy();ey=a['engine'].copy()
      S,D,z,dz,gs,gd=core.components(b,x,y,w,ax,ey);merged=core.merge(b,gs,gd,coefficient);payload=dict(main_x=x,main_y=y,main_weights=w,main_logits=z.numpy(),domain_logits=dz.numpy(),engine_labels=ey,losses=np.array([S,D],np.float32))
      for tag,grads in [('S',gs),('D',gd),('applied',merged)]:
       for i,g in enumerate(grads):
        if g is not None:payload[f'{tag}_{i}']=g.numpy()
      if step==0:
       assert sha(first['arrays_path'])==first['arrays_sha256']
       with np.load(first['arrays_path'],allow_pickle=False) as reference:
        for k in reference.files:
         if k.startswith(('S_','D_')) or k in ['main_x','main_y','main_weights','main_logits','domain_logits','engine_labels']:assert np.array_equal(payload[k],reference[k]),k
        assert np.array_equal(payload['losses'],reference['losses'][:2])
      opt.apply_gradients(zip(merged,vs));assert int(opt.iterations)==step+1 and core.sig(b['bn'])==bn and all(np.isfinite(v.numpy()).all() for v in state+list(opt.variables))
      p=folder/f'{step+1:03}.npz';np.savez_compressed(p,**payload);history.append(dict(step=step+1,losses=payload['losses'].tolist(),main_tensor_sha256=want['phase_sha256'],receipt=receipt,aux_arrays_sha256=aux[step]['arrays_sha256'],arrays_path=str(p),arrays_sha256=sha(p)));print(json.dumps(dict(**current,S=float(S),D=float(D))),flush=True)
     reader.log.close();m.save_weights(folder/'final.weights.h5');save(folder/'final-state.npz',state);save(folder/'optimizer.npz',opt.variables);ms=core.sig(m.variables);os=core.sig(opt.variables)
     if condition=='control':assert ms==baseline['final_signature'] and core.sig(opt.variables[:32])==baseline['optimizer_signature'] and [h['losses'][0] for h in history]==[h['loss'] for h in baseline['history']]
     row=dict(repeat=repeat,folder=str(folder),history=history,initial_inventory=initial,final_inventory=inventory(state),trainable_inventory=inventory(vs),wake_indices=b['wake'],encoder_indices=b['encoder'],optimizer_inventory=inventory(opt.variables),model_signature=ms,state_signature=core.sig(state),optimizer_signature=os,bn_unchanged=True,optimizer_iterations=int(opt.iterations),files={name:sha(folder/name) for name in ['initial.weights.h5','final.weights.h5','initial-state.npz','final-state.npz','optimizer.npz']});repetitions.append(row)
    assert repetitions[0]['state_signature']==repetitions[1]['state_signature'] and repetitions[0]['optimizer_signature']==repetitions[1]['optimizer_signature']
    reports.append(dict(target=target,condition=condition,coefficient=coefficient,repeats=repetitions))
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),cases=reports,gpu=core.factory.identity,all_passed=True,limits='Only16actualupdates, repeat/model/Adam/BN/input/gradient wiring smoke. No export/INT8/recognition/qualification/freshdata. Smoke weights never formal initializers.'),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());(OUT/'failure.json').write_text(json.dumps(dict(current=current,completed=reports,automatic_restart=False),indent=2));raise
if __name__=='__main__':main()
