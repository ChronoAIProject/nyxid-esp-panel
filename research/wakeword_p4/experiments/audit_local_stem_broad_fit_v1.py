"""Exhaustive admitted training-only broad bare support; no optimizer updates."""
import json,sys,subprocess,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx');LOCK=R/'4090-local-stem-broad-fit-plan-v1.json'
PHRASES=('Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def plan():
 p=json.loads((R/'4090-binary40-training-fit-plan-v1.json').read_text())
 for f in [Path(__file__),ROOT/'features.json',ROOT/'features.bin']:p['files'][str(f)]=sha(f)
 p['files']['.local/wakeword/binary40-local-stem-training-v1/final.pt']=sha('.local/wakeword/binary40-local-stem-training-v1/final.pt')
 for folder in ['experiments','training']:
  for f in Path('research/wakeword_p4',folder).glob('*.py'):p['files'][str(f)]=sha(f)
 p['policy']='Local-frequency stem candidate; All31644 original training broad bare rows in metadata order, five original classes, final20000weights, original features/masks and zero-cache startup, GPUfloat only, no fit/retry/dev/acceptance/threshold/calibration access. Full282calls preserve padded tails but label reporting separates valid source support. Existing training support, not independent generalization; no INT8/P4 equivalence.'
 assert not LOCK.exists();LOCK.write_text(json.dumps(p,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','0.2'],check=True)
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from causal_sequence_supervision_v1 import masks
 from audit_binary40_training_fit_v1 import crossing,float_events
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_local_stem_v1 import Binary40LocalStem
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load('.local/wakeword/binary40-local-stem-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
 m=Binary40LocalStem().cuda().eval();m.load_state_dict(cp['model'],strict=True);assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in m.state_dict().items())
 rows=[r for r in json.loads((ROOT/'features.json').read_text()) if r['source_pair_key']['carrier']=='bare' and r['text'] in PHRASES];assert len(rows)==31644
 assert all(r['split']=='training' and r['frames']<=846 for r in rows)
 raw=np.memmap(ROOT/'features.bin',dtype='<u2',mode='r').reshape(-1,40);prob=np.empty((len(rows),282),np.float32);records=[];groups={};start=time.monotonic()
 for off in range(0,len(rows),128):
  batch=rows[off:off+128];x=np.zeros((len(batch),846,40),np.float32);weights=[]
  for i,r in enumerate(batch):
   frames=r['frames'];x[i,:frames]=np.clip(raw[r['offset']:r['offset']+frames].astype(np.float32)/25.6,0,26)
   _,pw,nw=masks(frames,r['label'],int((r['lead_frame']+2)*160) if r['label'] else None,int(r['generated_container_end_sample']) if r['label'] else None);weights.append((pw,nw))
  with torch.inference_mode():
   y,_=m(torch.from_numpy(x).permute(0,2,1).cuda(),m.initial_state(len(batch),'cuda'));v=torch.sigmoid(y[:,0]).cpu().numpy();prob[off:off+len(batch)]=v
  for i,r in enumerate(batch):
   pw,nw=weights[i];z=v[i];events=float_events(z);valid=len(pw);cross=crossing(z[:valid],pw>0,False)
   g=groups.setdefault(r['engine']+'/'+r['text'],dict(rows=0,positive_crossing_rows=0,valid_event_rows=0,padded_only_event_rows=0));g['rows']+=1;g['positive_crossing_rows']+=bool(cross);g['valid_event_rows']+=any(s<=valid for s in events);g['padded_only_event_rows']+=bool(events) and all(s>valid for s in events)
   records.append(dict(index=off+i,offset=r['offset'],source_id=r['source_id'],variant=r['variant'],engine=r['engine'],conditioning_id=r['conditioning_id'],text=r['text'],label=r['label'],valid_calls=valid,events=events,positive_crossings=cross))
  if off%4096==0:print(json.dumps(dict(rows=off+len(batch),elapsed=time.monotonic()-start)),flush=True)
 trace=R/'4090-local-stem-broad-fit-traces-v1.npz';assert not trace.exists();np.savez_compressed(trace,probability=prob)
 out=dict(plan_sha256=sha(LOCK),groups=groups,records=records,trace=dict(path=str(trace),sha256=sha(trace)),elapsed_seconds=time.monotonic()-start,policy=p['policy'],optimizer_updates=0)
 dest=R/'4090-local-stem-broad-fit-original-v1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
