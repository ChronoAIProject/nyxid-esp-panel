"""Prospective actual-batch dual-branch gradient preflight, optimizer0."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-consistency-gradient-diagnostic-plan-v1.json';OUT=Path('.local/wakeword/hey-broad-consistency-gradient-diagnostic-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-paired-qat-formal-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for p in [parent,R/'4090-hey-broad-consistency-gradients-failure-v1.json',Path(__file__),Path(__file__).with_name('hey_broad_consistency_loss_v1.py'),Path(__file__).with_name('hey_broad_consistency_protocol_v1.md'),R/'4090-hey-broad-consistency-loss-check-v1.json',R/'4090-hey-broad-fixed-grid-recursive-independent-audit-v1.json',R/'4090-hey-broad-fixed-grid-head-decomposition-v1.json']:files[str(p)]=sha(p)
 snapshot=Path('.local/wakeword/hey-broad-consistency-gradients-v1/hey_willow/initial.weights.h5');files[str(snapshot)]=sha(snapshot)
 base['arms'][0]=dict(base['arms'][0]);assert base['arms'][0]['target']=='hey_willow';base['arms'][0]['path']=str(snapshot)
 with P.open('x') as f:json.dump(dict(files=files,versions=base['versions'],arms=base['arms'],export=base['export'],original_formal=base['original_formal'],batches=4,repeats=2,policy='Diagnostic only, retains original1e-5 component-gradient equality as measured flag, no admission/allpassed claim. Save all actual arrays and repeat arrays before assertions. Original Willow same-instance snapshot, Nyx original equalized snapshot. Originalfirstfour actual batches pertarget, originalequalizedinit/allBNfrozen. Both loss branches all15shared variables, save initializedH5/inventory and allactual gradients. Optimizer0; no new data or candidate admission. Lambda0/1 never adapted from gradient norms.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();lock=json.loads(P.read_text())
 for f,h in lock['files'].items():assert sha(f)==h,f
 import importlib.metadata
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_folded_operators_v1 import operators as floating
 from hey_broad_integer_encoder_training_v1 import operators as integer,configuration
 from hey_broad_consistency_loss_v1 import components
 from schedule_hey_broad_v1 import sampler
 from hey_batch_schedule_v1 import batch_digest
 from run_hey_broad_paired_qat_v1 import inventory
 tf=helper.tf;original=json.loads(Path(lock['original_formal']).read_text());exports=json.loads(Path(lock['export']).read_text())['arms'];OUT.mkdir(exist_ok=False);cases=[]
 for arm in lock['arms']:
  target=arm['target'];folder=OUT/target;folder.mkdir();model=helper.create(225);model.load_weights(arm['path'])
  for layer in model.layers:
   if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
  variables=model.trainable_variables;assert len(variables)==15 and all('GPU:0' in v.value.device for v in variables);before=[v.numpy().copy() for v in model.variables]
  model.save_weights(folder/'initial.weights.h5');(folder/'inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n')
  finfer=floating(model,tf);qinfer=integer(model,configuration(next(a for a in exports if a['target']==target)),tf)
  @tf.function
  def evaluate(x,y,w):
   with tf.GradientTape(persistent=True) as tape:
    zf=finfer(x);zq=qinfer(x)[0];total,s,c=components(zf,zq,y,w,1,tf)
   gradients=[tape.gradient(loss,variables) for loss in [s,c,total]]
   assert all(g is not None for group in gradients for g in group)
   for group in gradients:
    for g in group:tf.debugging.assert_all_finite(g,'nonfinite gradient')
   for value in [s,c,total]:tf.debugging.assert_all_finite(value,'nonfinite loss')
   return [s,c,total],gradients
  expected=[json.loads(l) for l in Path(original['arms'][target]['schedule']).read_text().splitlines()][:4];env=sampler(target)
  for batch,wanted in enumerate(expected,1):
   x,y,w=env['batch'](mine=False,augment=False);digest=batch_digest(x,y,w);assert digest==wanted['tensor_sha256']
   repeats=[]
   for _ in range(2):
    values,gradients=evaluate(tf.constant(x),tf.constant(y),tf.constant(w));repeats.append(([v.numpy() for v in values],[[g.numpy() for g in group] for group in gradients]))
   repeat_exact=all(np.array_equal(a,b) for a,b in zip(repeats[0][0],repeats[1][0])) and all(np.array_equal(a,b) for ga,gb in zip(repeats[0][1],repeats[1][1]) for a,b in zip(ga,gb))
   s,c,total=repeats[0][1]
   comparison=[]
   for j,(a,b,t) in enumerate(zip(s,c,total)):
    merged=a+b;error=np.abs(merged.astype(np.float64)-t);limit=1e-5+1e-5*np.abs(t.astype(np.float64));comparison.append(dict(variable=j,maximum_abs=float(error.max()),outside_original_tolerance=int(np.sum(error>limit)),values=int(error.size),merged_norm=float(np.linalg.norm(merged.astype(np.float64))),direct_norm=float(np.linalg.norm(t.astype(np.float64))),difference_norm=float(np.linalg.norm(error))))
   assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
   path=folder/f'batch{batch}.npz';arrays={f'{name}_{j}':v for name,group in zip(['supervised','consistency','total'],[s,c,total]) for j,v in enumerate(group)};arrays.update({f'repeat_{name}_{j}':v for name,group in zip(['supervised','consistency','total'],repeats[1][1]) for j,v in enumerate(group)});np.savez_compressed(path,x=x,y=y,w=w,losses=np.asarray(repeats[0][0]),repeat_losses=np.asarray(repeats[1][0]),**arrays)
   assert repeat_exact
   norm=lambda group:float(np.sqrt(sum(np.sum(v.astype(np.float64)**2) for v in group)))
   sn,cn=norm(s),norm(c);dot=float(sum(np.sum(a.astype(np.float64)*b.astype(np.float64)) for a,b in zip(s,c)));row=dict(target=target,batch=batch,batch_sha256=digest,losses=[float(v) for v in repeats[0][0]],supervised_gradient_norm=sn,consistency_gradient_norm=cn,cosine=dot/(sn*cn) if sn*cn else None,repeat_exact=repeat_exact,model_unchanged=True,component_comparison=comparison,original_additivity_tolerance_passed=all(v["outside_original_tolerance"]==0 for v in comparison),arrays_path=str(path),arrays_sha256=sha(path));cases.append(row);print(json.dumps(row),flush=True)
 result=dict(plan_sha256=sha(P),gpu=helper.identity,cases=cases,optimizer_updates=0,diagnostic_complete=len(cases)==8,original_additivity_all_passed=all(c["original_additivity_tolerance_passed"] for c in cases),limits=lock['policy'])
 with (R/'4090-hey-broad-consistency-gradient-diagnostic-result-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
