"""Authenticate terminal SAM artifacts and independently reconstruct all main batches."""
import gzip,json,math
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-cross-engine-carrier-training-v1/joint');LOCK=R/'4090-binary40-cross-engine-carrier-training-plan-v1.json'
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt_path=R/'4090-binary40-cross-engine-carrier-training-receipt-v1.json';receipt=json.loads(receipt_path.read_text());assert receipt['exit_code']==0
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 report_path=R/'4090-binary40-cross-engine-carrier-joint-training-original-v1.json';report=json.loads(report_path.read_text())
 assert report['steps']==20000 and report['arm']=='joint' and report['plan_sha256']==sha(LOCK) and report['launch']['pid']==receipt['pid']
 for p,h in report['files'].items():assert sha(B/p)==h,p
 checkpoints={}
 for name,step in [('initial',0),('final',20000)]:
  cp=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=False)
  assert cp['step']==step and cp['arm']=='joint' and cp['plan_sha256']==sha(LOCK)
  assert all(torch.isfinite(v).all() for v in cp['model'].values())
  assert cp['geometry']['parameters']==30593
  if step==0:assert not cp['optimizer']['state']
  else:
   states=cp['optimizer']['state'];assert len(states)==len(cp['model'])
   assert all(int(v['step'])==20000 for v in states.values())
   assert all(torch.isfinite(v[k]).all() for v in states.values() for k in ['exp_avg','exp_avg_sq'])
  checkpoints[name]=cp
 parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
 assert parent.keys()==checkpoints['initial']['model'].keys()
 assert all(torch.equal(v,parent[k]) for k,v in checkpoints['initial']['model'].items())
 from causal_source_balanced_batch_v1 import Sampler,digest
 sampler=Sampler(726);steps=0
 from causal_cross_engine_carrier_batch_v1 import Sampler as Treatment
 import numpy as np
 from causal_sequence_supervision_v1 import masks
 treatment=Treatment(726);rng=np.random.default_rng(242726)
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as baseline,gzip.open(B/'batches.jsonl.gz','rt') as current:
  while True:
   a=baseline.readline();b=current.readline()
   if not a and not b:break
   assert a and b
   a=json.loads(a);b=json.loads(b);steps+=1
   assert a['step']==b['step']==steps
   arrays,descriptors=sampler.batch()
   assert a['descriptors']==descriptors and b['descriptors'][:128]==descriptors and len(b['descriptors'])==136
   for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],arrays,strict=True):assert a[key]==digest(value)
   xx=[];pp=[];nn=[];extra=[]
   for pair in range(4):
    engine=['piper','kokoro'][int(rng.integers(2))];voices=treatment.voices[engine];voice=voices[int(rng.integers(len(voices)))];pool=treatment.pools[engine][voice];positive=pool[int(rng.integers(27))];k=positive['source_pair_key'];negative=treatment.negatives[engine][voice,float(k['scale']),k['carrier'],positive['variant']]
    for row,label in [(positive,1),(negative,0)]:
     frames=row['frames'];raw=treatment.arrays['positive' if label else engine][row['offset']:row['offset']+frames]
     x=np.zeros((846,40),np.float32);pw=np.zeros(282,np.float32);nw=pw.copy();x[:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
     _,pos,neg=masks(frames,label,(row['lead_frame']+2)*160 if label else None,row['generated_prefix_end_sample'] if label else None);pw[:len(pos)]=pos;nw[:len(neg)]=neg
     xx.append(x);pp.append(pw);nn.append(nw)
     extra.append(dict(kind='cross_engine_carrier_positive' if label else 'cross_engine_carrier_nick_negative',engine=engine,conditioning_id=voice,source_id=row.get('source_id',row.get('anchor_id')),index=row['offset'],carrier=k['carrier'],speed=float(k['scale']),variant=positive['variant'],label=label,pair=pair))
   assert b['descriptors']==descriptors+extra
   combined=tuple(np.concatenate([v,np.stack(w)],axis=0) for v,w in zip(arrays,[xx,pp,nn],strict=True))
   for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],combined,strict=True):assert b[key]==digest(value)
   assert b['batch_audit']==dict(original128_exact=True,appended_rows=8) and math.isfinite(b['loss'])
   if steps%2000==0:print(json.dumps(dict(verified_steps=steps)),flush=True)
 assert steps==20000
 out=dict(schema='binary40-cross-engine-carrier-training-review-v1',all_sampling_checks_passed=True,steps_verified=steps,descriptor_rows_verified=steps*136,initial_original_exact=True,all_initial_final_tensors_finite=True,optimizer_steps=20000,receipt_sha256=sha(receipt_path),report_sha256=sha(report_path),plan_sha256=sha(LOCK),reviewer_sha256=sha(__file__),recognition_scored=False,limits='Batch reconstruction and terminal artifact audit; all appended source tensors/masks/descriptors independently reconstructed; model optimizer arithmetic not replayed. No recognition/P4 claim.')
 with (R/'4090-binary40-cross-engine-carrier-training-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
