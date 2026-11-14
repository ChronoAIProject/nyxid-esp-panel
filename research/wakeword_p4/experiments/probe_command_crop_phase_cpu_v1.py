"""CPU final H5 nominal/phase crop vs saved stream at actual sampled ends."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-command-crop-phase-cpu-execution-v1.json';OUT=Path('.local/wakeword/command-crop-phase-cpu-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(action):
 if action=='lock':
  paths=[R/'4090-command-crop-phase-result-v1.json',R/'4090-command-training-endpoint-alignment-v1.json',Path(__file__),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),Path('.local/wakeword/hey-common-carrier-features-v1/features.json'),Path('.local/wakeword/hey-common-carrier-features-v1/features.bin'),Path('.local/wakeword/hey-common-carrier-training-schedules-v1/hey_nyx-command_context.jsonl'),Path('.local/wakeword/hey-common-carrier-raw-formal-v1/hey_nyx-command_context-0/final.weights.h5'),Path('.local/wakeword/common-carrier-command-training-streams-v1/records.jsonl')]
  paths+=sorted(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
  with LOCK.open('x') as f:json.dump(dict(files={str(p):sha(p) for p in paths},policy='All actual selected training draws with end divisible3 and end>=225, no score selection; repeated coordinates retained. Original225nominal crop vs224past+onezero phase crop, same finalH5/frozenBN, compare saved ownLiteRT stream output at exact end/3. Diagnostic only, not fitting or phase correction acceptance; no freshdev/acceptance.'),f,indent=2)
 else:
  assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
  for f,h in p['files'].items():assert sha(f)==h,f
  OUT.mkdir(exist_ok=False)
  import probe_hey_context_transfer_shapes_v1 as helper
  tf=helper.tf;assert not tf.config.get_visible_devices('GPU');tf.config.experimental.enable_op_determinism();model=helper.create(225);model.load_weights('.local/wakeword/hey-common-carrier-raw-formal-v1/hey_nyx-command_context-0/final.weights.h5')
  for layer in model.layers:
   cell=getattr(layer,'cell',layer)
   if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
  initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.variables)).hexdigest()
  meta=json.loads(Path('.local/wakeword/hey-common-carrier-features-v1/features.json').read_text());raw=np.memmap('.local/wakeword/hey-common-carrier-features-v1/features.bin',dtype='<u2',mode='r').reshape(-1,40)
  traces={}
  for line in Path('.local/wakeword/common-carrier-command-training-streams-v1/records.jsonl').read_text().splitlines():
   rec=json.loads(line)
   if rec['condition']=='command_context':
    assert sha(rec['trace'])==rec['trace_sha256']
    with np.load(rec['trace'],allow_pickle=False) as a:traces[rec['original_index']]=a['probability'].copy()
  requests=[]
  for line in Path('.local/wakeword/hey-common-carrier-training-schedules-v1/hey_nyx-command_context.jsonl').read_text().splitlines():
   batch=json.loads(line)
   for s in batch['selected']:
    if s['new_end']%3==0:requests.append(dict(step=batch['step'],**s))
  results=[]
  for start in range(0,len(requests),64):
   subset=requests[start:start+64];nominal=[];phase=[]
   for r in subset:
    row=meta[r['new_order']];end=r['new_end'];assert 225<=end<=row['frames'];at=row['offset'];x=np.clip(np.asarray(raw[at+end-225:at+end],dtype=np.float32)/25.6,0,26)
    nominal.append(x);phase.append(np.concatenate([x[1:],np.zeros((1,40),np.float32)]))
   n=model(np.asarray(nominal),training=False).numpy().reshape(-1);q=model(np.asarray(phase),training=False).numpy().reshape(-1)
   for r,a,b in zip(subset,n,q,strict=True):
    stream=float(traces[r['new_order']][r['new_end']//3-1]);results.append(dict(**r,nominal=float(a),phase=float(b),stream=stream,nominal_abs=abs(float(a)-stream),phase_abs=abs(float(b)-stream)))
   if start%1024==0:print(json.dumps(dict(completed=len(results),total=len(requests))),flush=True)
  assert initial==hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.variables)).hexdigest()
  assert results and all(np.isfinite(r[k]) for r in results for k in ('nominal','phase','stream'))
  report=dict(execution_sha256=sha(LOCK),runtime_device="CPU",visible_gpus=[g.name for g in tf.config.get_visible_devices("GPU")],requests=len(results),rows=results,weights_unchanged=True,nominal_abs={str(q):float(np.quantile([r['nominal_abs'] for r in results],q)) for q in (0,.5,.95,.99,1)},phase_abs={str(q):float(np.quantile([r['phase_abs'] for r in results],q)) for q in (0,.5,.95,.99,1)},threshold_selected=False,limits=p['policy'])
  with (R/'4090-command-crop-phase-cpu-result-v1.json').open('x') as f:json.dump(report,f,indent=2)
if __name__=='__main__':main(sys.argv[1])
