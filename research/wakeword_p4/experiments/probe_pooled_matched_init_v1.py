"""RTX4090 original-training-batch check of equivalent pooled/flat heads."""
import argparse,hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-pooled-matched-init-execution-v1.json';OUT=Path('.local/wakeword/pooled-matched-init-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-hey-broad-paired-qat-formal-execution-v1.json';p=json.loads(parent.read_text());phase=R/'4090-pooled-stream-phase-original-result-v1.json'
 models=[c for c in json.loads(phase.read_text())['cases'] if c['condition']=='float_control'];assert len(models)==2
 files=dict(p['files'])
 for f in [parent,phase,Path(__file__)]+[Path(c['h5_path']) for c in models]:files[str(f)]=sha(f)
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,models=models,original_formal=p['original_formal'],versions=p['versions'],batches=2,logit_max_abs=1e-4,contract='Primary original float-control encoders for Willow/Nyx, previously saved new pooled heads. Flatten head=tile(pooled W/30,30), bias identical. Initial encoder/BN exact, frozen BN. Same original first2 batches per target; save both initialized H5 before first forward, all batches/logits/losses/15gradients before assertions. Finite and max logit difference<=1e-4, model unchanged; optimizer0. No export/new media/dev. This is equivalence/wiring preflight, not training or INT8 admission. Future matched experiment fixed10000 final-only/Adam1e-5 same sampler must be separately locked; altered parameterization changes Adam dynamics, no sole-cause claim.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import importlib.metadata
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import hey_broad_gpu_training_factory_v1 as helper
  from microwakeword import mixednet
  from schedule_hey_broad_v1 import sampler
  from hey_batch_schedule_v1 import batch_digest
  tf=helper.tf;formal=json.loads(Path(p['original_formal']).read_text());rows=[]
  for m in p['models']:
   folder=OUT/m['target'];folder.mkdir();parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
   with tf.device('/GPU:0'):pool=mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None)
   pool.load_weights(m['h5_path']);flat=helper.create(225)
   enc=lambda model:[l for l in model.layers if l.weights and not isinstance(getattr(l,'cell',l),tf.keras.layers.Dense)]
   for a,b in zip(enc(pool),enc(flat),strict=True):
    assert type(getattr(a,'cell',a))==type(getattr(b,'cell',b));b.set_weights(a.get_weights())
    assert all(x.tobytes()==y.tobytes() for x,y in zip(a.get_weights(),b.get_weights(),strict=True))
   ph=pool.layers[-1];fh=flat.layers[-1];w,b=ph.get_weights();assert w.shape==(60,1)
   fh.set_weights([np.tile(w/np.float32(30),(30,1)),b.copy()])
   before={};embeddings={}
   for name,model in [('pool',pool),('flat',flat)]:
    for l in model.layers:
     if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
    assert len(model.trainable_variables)==15 and all('GPU:0' in v.value.device for v in model.trainable_variables)
    model.save_weights(folder/(name+'.initial.weights.h5'));before[name]=[v.numpy().copy() for v in model.variables]
    embeddings[name]=tf.keras.Model(model.inputs,model.layers[-1].input)
   env=sampler(m['target']);expected=[json.loads(l) for l in Path(formal['arms'][m['target']]['schedule']).read_text().splitlines()][:2]
   for i,wanted in enumerate(expected):
    x,y,weight=env['batch'](mine=False,augment=False);assert batch_digest(x,y,weight)==wanted['tensor_sha256'];saved=dict(x=x,y=y,weight=weight);values={}
    for name,model in [('pool',pool),('flat',flat)]:
     with tf.GradientTape() as tape:
      h=model.layers[-1];logits=tf.linalg.matmul(embeddings[name](tf.constant(x),training=False),h.kernel)+h.bias
      loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(y),logits=logits),[-1])*tf.constant(weight))
     gradients=tape.gradient(loss,model.trainable_variables);assert all(g is not None for g in gradients)
     saved[name+'_logits']=logits.numpy();saved[name+'_loss']=loss.numpy()
     for j,g in enumerate(gradients):saved[name+f'_gradient_{j:02}']=g.numpy()
     values[name]=logits.numpy()
    path=folder/f'batch{i:02}.npz';np.savez(path,**saved)
    finite=all(np.isfinite(v).all() for v in saved.values());gap=float(np.max(np.abs(values['pool'].astype(float)-values['flat'].astype(float))))
    row=dict(target=m['target'],batch=i,batch_sha256=wanted['tensor_sha256'],path=str(path),sha256=sha(path),finite=finite,logit_max_abs=gap,passed=bool(finite and gap<=p['logit_max_abs']));rows.append(row);print(json.dumps(row),flush=True)
   for name,model in [('pool',pool),('flat',flat)]:assert all(a.tobytes()==v.numpy().tobytes() for a,v in zip(before[name],model.variables,strict=True))
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),gpu=helper.identity,cases=rows,all_passed=all(r['passed'] for r in rows),optimizer_updates=0,limits=p['contract']),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
