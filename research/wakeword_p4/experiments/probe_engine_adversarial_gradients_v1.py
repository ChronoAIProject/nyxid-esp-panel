"""Actual4090 original-initialization, zero-optimizer engine-gradient preflight."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-adversarial-gradient-execution-v1.json';RESULT=R/'4090-engine-adversarial-gradient-result-v1.json';OUT=Path('.local/wakeword/engine-adversarial-gradient-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def signature(vs):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in vs)).hexdigest()
def inventory(vs):return [dict(index=i,name=str(getattr(v,'path',v.name)),shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for i,v in enumerate(vs)]
def state(path,vs):np.savez_compressed(path,**{f'v{i}':v.numpy() for i,v in enumerate(vs)})
def main():
 if sys.argv[1]=='lock':
  formal=R/'4090-frozen-encoder-formal-execution-v1.json';d=json.loads(formal.read_text());files=dict(d['files']);inp=R/'4090-engine-adversarial-input-result-v1.json';aud=R/'4090-engine-adversarial-input-independent-audit-v1.json';a=json.loads(aud.read_text());assert a['all_passed'] and a['original_result_sha256']==sha(inp)
  inputlock=R/'4090-engine-adversarial-input-execution-v1.json';files.update(json.loads(inputlock.read_text())['files'])
  for p in [formal,inp,aud,inputlock,Path(__file__),Path(__file__).with_name('engine_adversarial_gradient_probe_v1.md'),Path(__file__).with_name('audit_engine_adversarial_gradients_v1.py')]:files[str(p)]=sha(p)
  for batch in json.loads(inp.read_text())['batches']:files[batch['arrays_path']]=batch['arrays_sha256']
  for p,h in files.items():assert sha(p)==h,p
  with LOCK.open('x') as f:json.dump(dict(files=files,initial=d['initial'],schedules=d['schedules'],input_result=str(inp),domain_seed=20261118,batches=50,repeats=2,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']},policy='Original-init50main+50balancedaux batches per target, two tapes,0updates, reversal0/-1 mechanics. No efficacy/coeffselection/freshdata.'),f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 import hey_broad_gpu_training_factory_v1 as factory
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 from hey_batch_schedule_v1 import batch_digest
 tf=factory.tf;OUT.mkdir(exist_ok=False);reports=[];auxbatches=json.loads(Path(d['input_result']).read_text())['batches'];domain_identity=None
 def reverse(x,coefficient):
  @tf.custom_gradient
  def identity(a):
   return tf.identity(a),lambda dy:-coefficient*dy
  return identity(x)
 try:
  for target in ['hey_nyx','hey_willow']:
   folder=OUT/target;folder.mkdir();m=factory.create(225);m.load_weights(d['initial']['path']);head=m.layers[-1]
   for layer in m.layers:
    cell=getattr(layer,'cell',layer)
    if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
   vs=list(m.trainable_variables);assert len(vs)==15;wake=[i for i,v in enumerate(vs) if any(v is w for w in head.trainable_variables)];assert len(wake)==2;encoder=[i for i in range(15) if i not in wake];emb=tf.keras.Model(m.inputs,head.input)
   with tf.device('/GPU:0'):
    kernel=tf.Variable(tf.keras.initializers.GlorotUniform(seed=d['domain_seed'])((int(head.kernel.shape[0]),3),dtype=tf.float32),name='engine_kernel');bias=tf.Variable(tf.zeros([3],tf.float32),name='engine_bias')
   variables=vs+[kernel,bias];allstate=list(m.variables)+[kernel,bias];before=signature(allstate);inv=inventory(allstate);traininv=inventory(variables)
   if domain_identity is None:domain_identity=signature([kernel,bias])
   else:assert signature([kernel,bias])==domain_identity
   initialh5=folder/'initial.weights.h5';m.save_weights(initialh5);initialnpz=folder/'initial-variables.npz';state(initialnpz,allstate)
   (folder/'initialization.json').write_text(json.dumps(dict(model_initial_source=d['initial'],initial_h5_path=str(initialh5),initial_h5_sha256=sha(initialh5),state_path=str(initialnpz),state_sha256=sha(initialnpz),inventory=inv,trainable_inventory=traininv,wake_indices=wake,encoder_indices=encoder,domain_indices=[15,16]),indent=2)+'\n')
   reader=Batches(target,'command_context');schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];cases=[]
   for step in range(50):
    (x,y,w),receipt=reader.next();x=phase_input(x);want=schedule[step];assert receipt==want['original_receipt'] and batch_digest(x,y,w)==want['phase_sha256'];aux=auxbatches[step]
    with np.load(aux['arrays_path'],allow_pickle=False) as a:ax=a['x'].copy();ey=a['engine'].copy()
    saved=None;lossvalues=None;none=None
    for repeat in range(2):
     with tf.GradientTape(persistent=True) as tape:
      z=tf.linalg.matmul(emb(tf.constant(x),training=False),head.kernel)+head.bias
      S=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=z),[-1])*tf.constant(w))
      features=emb(tf.constant(ax),training=False);dz=tf.linalg.matmul(features,kernel)+bias;D=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ey,logits=dz))
      r0=tf.linalg.matmul(reverse(features,0.),kernel)+bias;r1=tf.linalg.matmul(reverse(features,1.),kernel)+bias
      L0=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ey,logits=r0));L1=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ey,logits=r1))
     gradients=[tape.gradient(loss,variables) for loss in [S,D,L0,L1]];del tape
     payload=dict(main_x=x,main_y=y,main_weights=w,main_logits=z.numpy(),domain_logits=dz.numpy(),engine_labels=ey,losses=np.asarray([S,D,L0,L1],np.float32));patterns=[]
     for tag,gs in zip(['S','D','R0','R1'],gradients):
      patterns.append([i for i,g in enumerate(gs) if g is None])
      for i,g in enumerate(gs):
       if g is not None:
        a=g.numpy();assert np.isfinite(a).all();payload[f'{tag}_{i}']=a
     assert patterns==[[15,16],wake,wake,wake]
     assert np.array_equal(dz.numpy(),r0.numpy()) and np.array_equal(dz.numpy(),r1.numpy()) and float(D)==float(L0)==float(L1)
     for i in encoder:assert np.count_nonzero(payload[f'R0_{i}'])==0 and np.array_equal(payload[f'R1_{i}'],-payload[f'D_{i}'])
     for i in [15,16]:assert np.array_equal(payload[f'R0_{i}'],payload[f'D_{i}']) and np.array_equal(payload[f'R1_{i}'],payload[f'D_{i}'])
     assert all(np.isfinite(v).all() for v in payload.values())
     if saved is None:saved=payload;lossvalues=payload['losses'].tolist();none=patterns
     else:assert saved.keys()==payload.keys() and all(np.array_equal(saved[k],payload[k]) for k in payload)
    assert signature(allstate)==before
    path=folder/f'{step+1:03}.npz';np.savez_compressed(path,**saved);case=dict(step=step+1,arrays_path=str(path),arrays_sha256=sha(path),main_tensor_sha256=want['phase_sha256'],original_receipt=receipt,aux_arrays_sha256=aux['arrays_sha256'],none_indices=none,losses=lossvalues,repeat_exact=True);cases.append(case)
    if (step+1)%10==0:print(json.dumps(dict(target=target,batches=step+1,optimizer_updates=0)),flush=True)
   reader.log.close();endnpz=folder/'end-variables.npz';state(endnpz,allstate);assert inventory(allstate)==inv and signature(allstate)==before
   reports.append(dict(target=target,cases=cases,initialization_path=str(folder/'initialization.json'),initialization_sha256=sha(folder/'initialization.json'),end_state_path=str(endnpz),end_state_sha256=sha(endnpz),all_state_unchanged=True,optimizer_updates=0))
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),gpu=factory.identity,targets=reports,all_passed=True,optimizer_created=False,limits=d['policy']),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
