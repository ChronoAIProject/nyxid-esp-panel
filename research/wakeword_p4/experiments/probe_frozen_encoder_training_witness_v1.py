"""Read-only final-model inference on authenticated, previously used training batches."""
import hashlib, importlib.metadata, json, subprocess, sys, traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
LOCK=R/'4090-frozen-encoder-training-witness-execution-v1.json'
RESULT=R/'4090-frozen-encoder-training-witness-result-v1.json'
OUT=Path('.local/wakeword/frozen-encoder-training-witness-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(m):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in m.variables)).hexdigest()
def make_infer(tf,emb,head):
 return tf.function(lambda x:tf.linalg.matmul(emb(x,training=False),head.kernel)+head.bias,input_signature=[tf.TensorSpec([128,225,40],tf.float32)])
def main():
 if sys.argv[1]=='lock':
  formal=R/'4090-frozen-encoder-formal-execution-v1.json';d=json.loads(formal.read_text());files=dict(d['files'])
  result=R/'4090-frozen-encoder-formal-result-v1.json';cases=json.loads(result.read_text())['cases']
  paths=[formal,result,R/'4090-frozen-encoder-formal-independent-audit-v1.json',Path(__file__),Path(__file__).with_name('frozen_encoder_training_witness_v1.md')]
  assert json.loads(paths[2].read_text())['all_passed']
  for c in cases:paths.append(Path(c['weights_path']));assert sha(c['weights_path'])==c['weights_sha256']
  for p in paths:files[str(p)]=sha(p)
  for p,h in files.items():assert sha(p)==h,p
  with LOCK.open('x') as f:json.dump(dict(files=files,cases=cases,schedules=d['schedules'],batches=100,repeats=2,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']},policy='Read-only first100 authenticated seen training batches, fixedfinal10000 only. Not generalization/recognition/INT8/acceptance evidence. No update or threshold selection.'),f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 import hey_broad_gpu_training_factory_v1 as h
 from common_carrier_training_batches_v1 import Batches
 from phase_aligned_batches_v1 import phase_input
 from hey_batch_schedule_v1 import batch_digest
 OUT.mkdir(exist_ok=False);cases=[]
 try:
  for target in ['hey_nyx','hey_willow']:
   sources=[c for c in d['cases'] if c['target']==target];models=[]
   for c in sources:
    m=h.create(225);m.load_weights(c['weights_path']);head=m.layers[-1]
    for l in m.layers:
     cell=getattr(l,'cell',l)
     if isinstance(cell,h.tf.keras.layers.BatchNormalization):l.trainable=cell.trainable=False
    emb=h.tf.keras.Model(m.inputs,head.input)
    infer=make_infer(h.tf,emb,head)
    models.append((c,m,infer,signature(m),[]))
   reader=Batches(target,'command_context');schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];receipts=[];ys=[];ws=[]
   for i in range(d['batches']):
    (x,y,w),receipt=reader.next();x=phase_input(x);want=schedule[i]
    assert receipt==want['original_receipt'] and batch_digest(x,y,w)==want['phase_sha256']
    assert np.count_nonzero(y==1)==32 and np.count_nonzero(y==0)==96
    assert np.array_equal(w,np.where(y[:,0]==1,1,20))
    receipts.append(dict(step=i+1,tensor_sha256=want['phase_sha256'],receipt=receipt));ys.append(y[:,0]);ws.append(w)
    for c,m,infer,sig,values in models:
     a=infer(x).numpy()[:,0];b=infer(x).numpy()[:,0];assert np.array_equal(a,b) and np.isfinite(a).all();values.append(a)
    if (i+1)%10==0:print(json.dumps(dict(target=target,batches=i+1)),flush=True)
   reader.log.close() # Deliberately bounded100 prefix; do not claim finish()10000 audit.
   rp=OUT/f'{target}-receipts.json';rp.write_text(json.dumps(receipts,indent=2)+'\n')
   for c,m,infer,sig,values in models:
    assert signature(m)==sig and sha(c['weights_path'])==c['weights_sha256']
    logits=np.stack(values);p=h.tf.math.sigmoid(logits).numpy();assert np.isfinite(p).all()
    path=OUT/f"{target}-{c['condition']}.npz";np.savez_compressed(path,logits=logits,probabilities=p,labels=np.stack(ys),weights=np.stack(ws))
    cases.append(dict(target=target,condition=c['condition'],weights_sha256=c['weights_sha256'],arrays_path=str(path),arrays_sha256=sha(path),receipts_path=str(rp),receipts_sha256=sha(rp),rows=12800,repeat_exact=True,model_unchanged=True))
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),gpu=h.identity,cases=cases,all_passed=True,limits=d['policy']),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
