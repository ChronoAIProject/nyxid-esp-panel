"""Single fixed-final four-arm engine-adversarial run. Never resumes or retries."""
import datetime,gzip,hashlib,importlib.metadata,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-adversarial-formal-execution-v1.json';RESULT=R/'4090-engine-adversarial-formal-result-v1.json';OUT=Path('.local/wakeword/engine-adversarial-formal-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def receipt_sha(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def inventory(vs):return [dict(index=i,name=str(getattr(v,'path',v.name)),shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for i,v in enumerate(vs)]
def save(p,vs):np.savez_compressed(p,**{f'v{i}':v.numpy() for i,v in enumerate(vs)})
def main():
 if sys.argv[1]=='lock':
  parent=R/'4090-engine-adversarial-smoke-execution-v1.json';d=json.loads(parent.read_text());files=dict(d['files']);smoke=R/'4090-engine-adversarial-smoke-result-v1.json';audit=R/'4090-engine-adversarial-smoke-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_passed'] and a['source_result_sha256']==sha(smoke)
  schedule=R/'4090-engine-adversarial-schedule-result-v1.json';sa=R/'4090-engine-adversarial-schedule-independent-audit-v1.json';sr=json.loads(schedule.read_text());ar=json.loads(sa.read_text());assert sr['all_passed'] and ar['all_passed'] and ar['source_result_sha256']==sha(schedule)
  files.update(json.loads((R/'4090-engine-adversarial-schedule-execution-v1.json').read_text())['files'])
  old=R/'4090-frozen-encoder-formal-result-v1.json';old_audit=R/'4090-frozen-encoder-formal-independent-audit-v1.json';assert json.loads(old_audit.read_text())['all_passed']
  paths=[parent,smoke,audit,schedule,sa,R/'4090-engine-adversarial-schedule-execution-v1.json',Path(sr['path']),old,old_audit,Path(__file__),Path(__file__).with_name('engine_adversarial_formal_v1.md'),Path(__file__).with_name('audit_engine_adversarial_formal_v1.py')]
  for c in json.loads(old.read_text())['cases']:
   if c['condition']=='full':paths.extend([Path(c['weights_path']),Path(c['optimizer_path'])])
  for p in paths:files[str(p)]=sha(p)
  for p,h in files.items():assert sha(p)==h,p
  d.update(files=files,smoke_result=str(smoke),auxiliary_schedule=sr,old_formal=str(old),steps=10000,repeats=1,targets=['hey_nyx','hey_willow'],versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']})
  with LOCK.open('x') as f:json.dump(d,f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 import engine_adversarial_training_core_v1 as core
 from engine_adversarial_batches_v1 import AuxiliaryBatches
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 from hey_batch_schedule_v1 import batch_digest
 tf=core.tf;OUT.mkdir(exist_ok=False);startup=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_sha256=sha(LOCK),automatic_restart=False);(OUT/'startup.json').write_text(json.dumps(startup,indent=2)+'\n');reports=[];current=None;smoke=json.loads(Path(d['smoke_result']).read_text());old=json.loads(Path(d['old_formal']).read_text())
 try:
  for target in d['targets']:
   schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];assert len(schedule)==10000;baseline=next(c for c in old['cases'] if c['target']==target and c['condition']=='full')
   for condition,coefficient in d['coefficient'].items():
    folder=OUT/f'{target}-{condition}';folder.mkdir();current=dict(target=target,condition=condition,step=0);expected=next(c for c in smoke['cases'] if c['target']==target and c['condition']==condition)['repeats'][0];b=core.make(d['initial']['path'],d['domain_seed']);m=b['model'];vs=b['variables'];state=b['state'];bn=core.sig(b['bn']);initial=inventory(state);assert [v['sha256'] for v in initial]==[v['sha256'] for v in expected['initial_inventory']];m.save_weights(folder/'initial.weights.h5');save(folder/'initial-state.npz',state);(folder/'initialization.json').write_text(json.dumps(dict(initial_inventory=initial,trainable_inventory=inventory(vs),wake_indices=b['wake'],encoder_indices=b['encoder']),indent=2)+'\n');opt=tf.keras.optimizers.Adam(1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7);opt.build(vs);assert len(opt.variables)==36 and int(opt.iterations)==0 and all(np.count_nonzero(v.numpy())==0 for v in opt.variables[2:]);save(folder/'initial-optimizer.npz',opt.variables);reader=Batches(target,'command_context');aux=AuxiliaryBatches();snapshots=[];started=time.monotonic()
    with gzip.open(d['auxiliary_schedule']['path'],'rt') as auxlog,(folder/'history.jsonl').open('x') as log:
     for step in range(1,10001):
      current['step']=step;(x,y,w),receipt=reader.next();x=phase_input(x);want=schedule[step-1];assert receipt==want['original_receipt'] and batch_digest(x,y,w)==want['phase_sha256'];(ax,ey,ty),auxreceipt=aux.next();assert auxreceipt==json.loads(next(auxlog))
      S,D,z,dz,gs,gd=core.components(b,x,y,w,ax,ey);merged=core.merge(b,gs,gd,coefficient);losses=[float(S),float(D)]
      if step<=2:assert losses==expected['history'][step-1]['losses'] and want['phase_sha256']==expected['history'][step-1]['main_tensor_sha256']
      opt.apply_gradients(zip(merged,vs));assert int(opt.iterations)==step and core.sig(b['bn'])==bn and all(np.isfinite(v.numpy()).all() for v in state+list(opt.variables))
      if step==2:assert core.sig(state)==expected['state_signature'] and core.sig(opt.variables)==expected['optimizer_signature']
      row=dict(step=step,losses=losses,main_tensor_sha256=want['phase_sha256'],main_receipt_sha256=receipt_sha(receipt),aux_input_sha256=auxreceipt['input_sha256'],aux_receipt_sha256=receipt_sha(auxreceipt),optimizer_iteration=int(opt.iterations));log.write(json.dumps(row)+'\n');log.flush()
      if step%1000==0:
       prefix=f'step{step}';m.save_weights(folder/(prefix+'.weights.h5'));save(folder/(prefix+'-state.npz'),state);save(folder/(prefix+'-optimizer.npz'),opt.variables);snapshots.append(dict(step=step,files={name:sha(folder/name) for name in [prefix+'.weights.h5',prefix+'-state.npz',prefix+'-optimizer.npz']}))
      if step<=2 or step%100==0:print(json.dumps(dict(target=target,condition=condition,**row)),flush=True)
     assert auxlog.readline()==''
    reader.finish();assert aux.count==10000
    if condition=='control':assert core.sig(m.variables)==baseline['final_signature'] and core.sig(opt.variables[:32])==baseline['optimizer_signature']
    m.save_weights(folder/'final.weights.h5');save(folder/'final-state.npz',state);save(folder/'optimizer.npz',opt.variables)
    report=dict(target=target,condition=condition,coefficient=coefficient,steps=10000,folder=str(folder),elapsed_seconds=time.monotonic()-started,history_sha256=sha(folder/'history.jsonl'),initial_inventory=initial,final_inventory=inventory(state),optimizer_inventory=inventory(opt.variables),model_signature=core.sig(m.variables),state_signature=core.sig(state),optimizer_signature=core.sig(opt.variables),bn_unchanged=True,optimizer_iterations=int(opt.iterations),snapshots=snapshots,files={name:sha(folder/name) for name in ['initial.weights.h5','initial-state.npz','initial-optimizer.npz','initialization.json','final.weights.h5','final-state.npz','optimizer.npz']});(folder/'training.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),startup=startup,cases=reports,gpu=core.factory.identity,all_passed=True,limits='Fourfixedfinal10000training runs only, no export/numeric/recognition/INT8 or promotion claim.'),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());(OUT/'failure.json').write_text(json.dumps(dict(current=current,completed=reports,automatic_restart=False),indent=2));raise
if __name__=='__main__':main()
