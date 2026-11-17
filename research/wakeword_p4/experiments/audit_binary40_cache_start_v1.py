"""Isolate cache initialization using unchanged training features, not PCM replay."""
import gzip,json,subprocess,sys
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-cache-start-plan-v1.json'
def plan():
 p=json.loads((R/'4090-binary40-training-fit-plan-v1.json').read_text());p['files'][str(Path(__file__))]=sha(__file__)
 p['policy']='Original Sampler726 first128, strict final weights; compare zero cache with cache after complete 81600zeroPCM frontend features, consuming only complete threeframe calls (508 frames ->507, leave1 unconsumed). Unchanged original tensors/masks, reset event decoder on real tensor in both arms. Isolates neural cache only: not continuous frontend/physical waveform, not deployment event carry, no new training or acceptance.'
 LOCK.write_text(json.dumps(p,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from causal_source_balanced_batch_v1 import Sampler,digest
 from audit_binary40_training_fit_v1 import crossing,float_events
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from features import features
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
 m=Binary40().cuda().eval();m.load_state_dict(cp['model'],strict=True);assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in m.state_dict().items())
 (x,pw,nw),desc=Sampler(726).batch()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:original=json.loads(next(f))
 assert desc==original['descriptors']
 for key,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(a)==original[key]
 raw=features(np.zeros(81600,np.int16));prefix=np.clip(raw[:len(raw)//3*3].astype(np.float32)/25.6,0,26)
 with torch.inference_mode():
  t=torch.from_numpy(x).permute(0,2,1).cuda();cold,_=m(t,m.initial_state(128,'cuda'))
  _,state=m(torch.from_numpy(np.repeat(prefix.T[None],128,axis=0)).cuda(),m.initial_state(128,'cuda'))
  warm,_=m(t,state);cold=torch.sigmoid(cold[:,0]).cpu().numpy();warm=torch.sigmoid(warm[:,0]).cpu().numpy()
 saved=json.loads((R/'4090-binary40-training-fit-original-v1.json').read_text())
 with np.load(saved['trace']['path']) as z:assert np.array_equal(cold,z['float_probability'])
 rows=[]
 for i,d in enumerate(desc):
  rows.append(dict(index=i,source=d,positive=bool(pw[i].sum()),cold_crossings=crossing(cold[i],pw[i]>0,False),warm_crossings=crossing(warm[i],pw[i]>0,False),cold_events=float_events(cold[i]),warm_events=float_events(warm[i]),max_positive_difference=float(np.max(np.abs(cold[i]-warm[i])[pw[i]>0],initial=0)),max_negative_difference=float(np.max(np.abs(cold[i]-warm[i])[nw[i]>0],initial=0))))
 trace=R/'4090-binary40-cache-start-traces-v1.npz';assert not trace.exists();np.savez_compressed(trace,cold=cold,warm=warm,positive_weights=pw,negative_weights=nw,prefix=prefix)
 summary=dict(cold_positive_rows=sum(bool(r['cold_crossings']) for r in rows),warm_positive_rows=sum(bool(r['warm_crossings']) for r in rows),cold_negative_event_rows=sum(bool(r['cold_events']) for r in rows if not r['positive']),warm_negative_event_rows=sum(bool(r['warm_events']) for r in rows if not r['positive']),changed_event_rows=[r['index'] for r in rows if r['cold_events']!=r['warm_events']],max_positive_difference=max(r['max_positive_difference'] for r in rows),max_negative_difference=max(r['max_negative_difference'] for r in rows))
 out=dict(plan_sha256=sha(LOCK),summary=summary,records=rows,raw_silence_frames=len(raw),prefix_frames=len(prefix),trace=dict(path=str(trace),sha256=sha(trace)),policy=p['policy'],optimizer_updates=0)
 dest=R/'4090-binary40-cache-start-original-v1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
