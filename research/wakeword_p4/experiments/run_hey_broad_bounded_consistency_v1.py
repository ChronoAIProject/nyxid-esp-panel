"""Locked matched GPU fine-tuning, disposable smoke or fixed-final formal run."""
import argparse,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(variables):return [dict(shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in variables]
def signature(variables):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in variables)).hexdigest()
def plan(mode):
 assert mode=='smoke'
 parent=R/'4090-hey-broad-paired-qat-formal-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for f in [parent,Path(__file__),Path(__file__).with_name('hey_broad_bounded_consistency_preflight_v1.md'),Path(__file__).with_name('hey_broad_bounded_consistency_smoke_v1.md'),Path(__file__).with_name('hey_broad_consistency_loss_v1.py'),Path(__file__).with_name('hey_broad_bounded_consistency_loss_v1.py'),Path(__file__).with_name('hey_broad_scalar_fc_training_observer_v1.py'),Path(__file__).with_name('hey_broad_scalar_fc_ste_v1.py'),Path(__file__).with_name('hey_broad_consistency_gradient_merge_v1.py'),R/'4090-bounded-consistency-gradients-result-v1.json',R/'4090-bounded-consistency-gradients-independent-audit-v1.json']:files[str(f)]=sha(f)
 assert len(json.loads((R/'4090-bounded-consistency-gradients-result-v1.json').read_text())['cases'])==8
 assert len(json.loads((R/'4090-bounded-consistency-gradients-independent-audit-v1.json').read_text())['cases'])==8
 lock=dict(schema='hey-broad-bounded-consistency-execution-v1',mode=mode,files=files,versions=base['versions'],arms=base['arms'],export=base['export'],original_formal=base['original_formal'],steps=4 if mode=='smoke' else 10000,repeats=2 if mode=='smoke' else 1,optimizer=base['optimizer'],conditions=['raw_logit_control','bounded_logit_consistency'],output=f'.local/wakeword/hey-broad-bounded-consistency-{mode}-v1',policy='Newpolicy constant_folding=False/arithmetic_optimization=False. Both targets raw-C vs nominalbounded-C, fixedlambda1; exactFC observed in both. Fixed dualbranch S+lambdaC, explicit separately differentiated gradients merged, raw-vs-boundedC fixedlambda1. Same originalequalizedinit/frozenBN/first10000batchSHA, finalonly. Smoke weights never formal. No newdata/dev/acceptance/export/promotion. Original direct-gradient failure retained, no silent retries.')
 dest=R/f'4090-hey-broad-bounded-consistency-{mode}-execution-v1.json'
 with dest.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
def run(mode):
 assert mode=='smoke'
 p=R/f'4090-hey-broad-bounded-consistency-{mode}-execution-v1.json';assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)])==p.read_bytes();lock=json.loads(p.read_text())
 for f,h in lock['files'].items():assert sha(f)==h,f
 import importlib.metadata
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_folded_operators_v1 import operators as floating
 from hey_broad_scalar_fc_training_observer_v1 import operators as quantized,configuration
 from hey_broad_consistency_loss_v1 import components as raw_components
 from hey_broad_bounded_consistency_loss_v1 import components as bounded_components
 from hey_broad_consistency_gradient_merge_v1 import merge
 from schedule_hey_broad_v1 import sampler
 from hey_batch_schedule_v1 import batch_digest
 tf=helper.tf;tf.config.optimizer.set_experimental_options({'constant_folding':False,'arithmetic_optimization':False});original=json.loads(Path(lock['original_formal']).read_text());exports=json.loads(Path(lock['export']).read_text())['arms'];base=Path(lock['output']);base.mkdir(exist_ok=False);cases=[]
 for arm in lock['arms']:
  target=arm['target'];expected=[json.loads(l) for l in Path(original['arms'][target]['schedule']).read_text().splitlines()][:lock['steps']];assert len(expected)==lock['steps']
  for condition in lock['conditions']:
   identities=[]
   for repeat in range(lock['repeats']):
    out=base/f'{target}-{condition}-{repeat}';out.mkdir();model=helper.create(225);model.load_weights(arm['path'])
    for layer in model.layers:
     c=getattr(layer,'cell',layer)
     if isinstance(c,tf.keras.layers.BatchNormalization):layer.trainable=c.trainable=False
    variables=model.trainable_variables;assert len(variables)==15 and all('GPU:0' in v.value.device for v in variables);initial=[v.numpy().copy() for v in variables];frozen=[v.numpy().copy() for v in model.non_trainable_variables];model.save_weights(out/'initial.weights.h5');(out/'initial-inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n')
    optimizer=tf.keras.optimizers.Adam(**lock['optimizer']);optimizer.build(variables)
    finfer=floating(model,tf);qinfer=quantized(model,configuration(next(a for a in exports if a['target']==target)),tf);bounded=condition=='bounded_logit_consistency'
    @tf.function
    def step(x,y,weight):
     with tf.GradientTape(persistent=True) as tape:
      zf=finfer(x);zq,prob,observed=qinfer(x);bf=tf.clip_by_value(zf,-8.,8.);bq=tf.clip_by_value((tf.cast(observed['scalar_fc_exact'],tf.float32)-3)*tf.constant(8/3,tf.float32),-8.,8.)
      raw_total,supervised,raw_c=raw_components(zf,zq,y,weight,1,tf);bounded_total,bounded_s,bounded_c=bounded_components(zf,zq,bf,bq,y,weight,1,tf);tf.debugging.assert_equal(supervised,bounded_s)
      loss=bounded_total if bounded else raw_total;chosen=bounded_c if bounded else raw_c
     sg=tape.gradient(supervised,variables);raw_cg=tape.gradient(raw_c,variables);bounded_cg=tape.gradient(bounded_c,variables);cg=bounded_cg if bounded else raw_cg;grads=merge(sg,cg,1,tf)
     tf.debugging.assert_all_finite(loss,'Nonfinite loss')
     optimizer.apply_gradients(zip(grads,variables))
     return loss,tf.linalg.global_norm(grads),supervised,chosen,tf.linalg.global_norm(sg),tf.linalg.global_norm(cg),raw_c,bounded_c,tf.linalg.global_norm(raw_cg),tf.linalg.global_norm(bounded_cg)
    env=sampler(target);start=time.monotonic();history=[]
    with (out/'actual-batch-schedule.jsonl').open('x') as trace:
     for index,wanted in enumerate(expected,1):
      x,y,w=env['batch'](mine=False,augment=False);digest=batch_digest(x,y,w);assert digest==wanted['tensor_sha256'],(target,condition,index)
      loss,norm,sl,cl,sn,cn,rc,bc,rn,bn=step(tf.constant(x),tf.constant(y),tf.constant(w));assert all(np.isfinite(v.numpy()).all() for v in variables);assert all(np.array_equal(a,v.numpy()) for a,v in zip(frozen,model.non_trainable_variables,strict=True));assert int(optimizer.iterations)==index
      row=dict(step=index,batch_sha256=digest,loss=float(loss),gradient_l2=float(norm),supervised_loss=float(sl),consistency_loss=float(cl),supervised_gradient_l2=float(sn),consistency_gradient_l2=float(cn),raw_consistency_loss=float(rc),bounded_consistency_loss=float(bc),raw_consistency_gradient_l2=float(rn),bounded_consistency_gradient_l2=float(bn));trace.write(json.dumps(row)+'\n');trace.flush()
      if index<=2 or index%1000==0 or index==lock['steps']:
       history.append(row);print(json.dumps(dict(target=target,condition=condition,repeat=repeat,**row)),flush=True)
      if mode=='formal' and index%1000==0:
       model.save_weights(out/f'step{index}.weights.h5');tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(out/f'step{index}'))
    delta=float(np.sqrt(sum(np.sum((v.numpy().astype(float)-a.astype(float))**2) for a,v in zip(initial,variables,strict=True))));assert delta>0
    model.save_weights(out/'final.weights.h5');np.savez_compressed(out/'final-optimizer.npz',**{f'v{n}':v.numpy() for n,v in enumerate(optimizer.variables)});(out/'final-inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n')
    identity=dict(model=signature(model.variables),optimizer=signature(optimizer.variables));identities.append(identity)
    receipt=dict(target=target,condition=condition,repeat=repeat,steps=lock['steps'],elapsed_seconds=time.monotonic()-start,parameter_delta_l2=delta,frozen_bn_exact=True,finite=True,identity=identity,weights_path=str(out/'final.weights.h5'),weights_sha256=sha(out/'final.weights.h5'),optimizer_path=str(out/'final-optimizer.npz'),optimizer_sha256=sha(out/'final-optimizer.npz'),batch_trace_sha256=sha(out/'actual-batch-schedule.jsonl'),history=history)
    (out/'training.json').write_text(json.dumps(receipt,indent=2)+'\n')
   assert all(i==identities[0] for i in identities)
   cases.append(dict(target=target,condition=condition,steps=lock['steps'],fresh_update_repeats_exact=True,final_repeat=receipt))
 result=dict(execution_sha256=sha(p),gpu=helper.identity,cases=cases,all_passed=len(cases)==4,limits=lock['policy'])
 dest=R/f'4090-hey-broad-bounded-consistency-{mode}-result-v1.json'
 with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('mode',choices=['smoke']);a=p.parse_args();plan(a.mode) if a.action=='plan' else run(a.mode)
