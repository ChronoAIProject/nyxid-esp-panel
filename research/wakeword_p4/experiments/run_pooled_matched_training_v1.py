"""Matched pool/flat raw-BCE training; adapted original audited paired runner."""
import argparse,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(variables):return [dict(shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in variables]
def signature(variables):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in variables)).hexdigest()
def plan(mode):
 parent=R/'4090-pooled-matched-init-execution-v1.json';p=json.loads(parent.read_text());files=dict(p['files']);formal=Path(p['original_formal'])
 init=R/'4090-pooled-matched-init-original-result-v1.json';audit=R/'4090-pooled-matched-init-saved-array-audit-v1.json'
 assert json.loads(init.read_text())['all_passed'] and json.loads(audit.read_text())['all_passed']
 for f in [parent,init,audit,Path(__file__)]:files[str(f)]=sha(f)
 initial=json.loads(audit.read_text())['initial_h5'];files.update(initial)
 if mode=='formal':
  smoke=R/'4090-pooled-matched-smoke-result-v1.json';assert json.loads(smoke.read_text())['all_passed'];files[str(smoke)]=sha(smoke)
  independent=R/'4090-pooled-matched-smoke-independent-audit-v1.json';assert json.loads(independent.read_text())['all_passed'];files[str(independent)]=sha(independent)
 original=json.loads(formal.read_text())
 for a in original['arms'].values():files[a['schedule']]=sha(a['schedule'])
 arms=[dict(target=m['target'],initial={c:'.local/wakeword/pooled-matched-init-v1/'+m['target']+'/'+c+'.initial.weights.h5' for c in ['flat','pool']}) for m in p['models']]
 lock=dict(schema='pooled-matched-execution-v1',mode=mode,files=files,versions=p['versions'],arms=arms,original_formal=str(formal),steps=2 if mode=='smoke' else 10000,repeats=2 if mode=='smoke' else 1,optimizer=dict(learning_rate=1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7,amsgrad=False),conditions=['flat','pool'],output=f'.local/wakeword/pooled-matched-{mode}-v1',policy='Two targets times flat/pool, equal initial function and encoder/BN, raw weighted BCE original labels/negative20. Original first10000 batch hashes, fresh Adam1e-5 all15trainables, frozenBN. Smoke2updates twice exact; formal10000 fixedfinal only, checkpoints evidence not selection. Pooling is inductive-bias/parameterization intervention including different Adam dynamics, not solely parameter count. No QAT/consistency/bounded rewrite/new data/dev. Full waveform INT8/recognition/FAR gates required later, v30 unchanged.')
 with (R/f'4090-pooled-matched-{mode}-execution-v1.json').open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
def run(mode):
 p=R/f'4090-pooled-matched-{mode}-execution-v1.json';assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)])==p.read_bytes();lock=json.loads(p.read_text())
 for f,h in lock['files'].items():assert sha(f)==h,f
 import importlib.metadata
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
 import hey_broad_gpu_training_factory_v1 as helper
 from microwakeword import mixednet
 from schedule_hey_broad_v1 import sampler
 from hey_batch_schedule_v1 import batch_digest
 tf=helper.tf;original=json.loads(Path(lock['original_formal']).read_text());base=Path(lock['output']);base.mkdir(exist_ok=False);cases=[]
 for arm in lock['arms']:
  target=arm['target'];expected=[json.loads(l) for l in Path(original['arms'][target]['schedule']).read_text().splitlines()][:lock['steps']];assert len(expected)==lock['steps']
  for condition in lock['conditions']:
   identities=[]
   for repeat in range(lock['repeats']):
    out=base/f'{target}-{condition}-{repeat}';out.mkdir()
    if condition=='flat':model=helper.create(225)
    else:
     parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
     with tf.device('/GPU:0'):model=mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None)
    model.load_weights(arm['initial'][condition])
    for layer in model.layers:
     c=getattr(layer,'cell',layer)
     if isinstance(c,tf.keras.layers.BatchNormalization):layer.trainable=c.trainable=False
    variables=model.trainable_variables;assert len(variables)==15 and all('GPU:0' in v.value.device for v in variables);initial=[v.numpy().copy() for v in variables];frozen=[v.numpy().copy() for v in model.non_trainable_variables];model.save_weights(out/'initial.weights.h5');(out/'initial-inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n')
    optimizer=tf.keras.optimizers.Adam(**lock['optimizer']);optimizer.build(variables)
    embedding=tf.keras.Model(model.inputs,model.layers[-1].input);head=model.layers[-1]
    def infer(x):return tf.linalg.matmul(embedding(x,training=False),head.kernel)+head.bias
    @tf.function
    def step(x,y,weight):
     with tf.GradientTape() as tape:
      value=infer(x);logits=value
      loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits),[-1])*weight)
     grads=tape.gradient(loss,variables);assert all(g is not None for g in grads)
     tf.debugging.assert_all_finite(loss,'Nonfinite loss')
     for g in grads:tf.debugging.assert_all_finite(g,'Nonfinite gradient')
     optimizer.apply_gradients(zip(grads,variables))
     return loss,tf.linalg.global_norm(grads)
    env=sampler(target);start=time.monotonic();history=[]
    with (out/'actual-batch-schedule.jsonl').open('x') as trace:
     for index,wanted in enumerate(expected,1):
      x,y,w=env['batch'](mine=False,augment=False);digest=batch_digest(x,y,w);assert digest==wanted['tensor_sha256'],(target,condition,index)
      loss,norm=step(tf.constant(x),tf.constant(y),tf.constant(w));assert all(np.isfinite(v.numpy()).all() for v in variables);assert all(np.array_equal(a,v.numpy()) for a,v in zip(frozen,model.non_trainable_variables,strict=True));assert int(optimizer.iterations)==index
      row=dict(step=index,batch_sha256=digest,loss=float(loss),gradient_l2=float(norm));trace.write(json.dumps(row)+'\n');trace.flush()
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
 dest=R/f'4090-pooled-matched-{mode}-result-v1.json'
 with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('mode',choices=['smoke','formal']);a=p.parse_args();plan(a.mode) if a.action=='plan' else run(a.mode)
