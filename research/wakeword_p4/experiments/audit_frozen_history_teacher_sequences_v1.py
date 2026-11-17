"""All original synthetic training presentations, same aligned masks and padding."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');P=R/'4090-frozen-history-teacher-sequence-fit-plan-v1.json'
CHECKPOINTS={'original':'.local/wakeword/binary40-training-v1/final.pt',**{a:f'.local/wakeword/frozen-history-teacher-training-v1/{a}/final.pt' for a in ['control','teacher']}}
def plan():
 material=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');rows=[json.loads(s) for s in material.read_text().splitlines()];assert len(rows)==51840
 files=dict(json.loads((R/'4090-binary40-training-plan-v1.json').read_text())['files'])
 paths=[Path(__file__),material,R/'4090-causal-command-prefix-reference-map-v1.json',*[Path(p) for p in CHECKPOINTS.values()],*[Path(p) for p in sorted({r['feature_binary'] for r in rows})]]
 for name in ['causal_aligned_prefix_batch_v1.py','causal_sequence_batch_v1.py','causal_sequence_supervision_v1.py','audit_binary40_training_fit_v1.py']:paths.append(Path(__file__).with_name(name))
 files.update({str(p):sha(p) for p in paths})
 with P.open('x') as f:json.dump(dict(files=files,checkpoints=CHECKPOINTS,rows=51840,padded_frames=846,policy='All51840original synthetic training presentations in material order, original aligned-prefix private by_index masks and main846padding; finalweights original/control/teacher GPUfloat, no optimizer or dev/acceptance sources. Save all282probabilities, supervised crossings and valid/padded events. Training support only, not physicalPCM/generalization/P4. No tuning/filtering/threshold rescue.'),f,indent=2)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from causal_aligned_prefix_batch_v1 import Sampler
 from causal_sequence_supervision_v1 import masks
 from audit_binary40_training_fit_v1 import crossing,float_events
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 sampler=Sampler(726);rows=[sampler.by_index[i] for i in range(51840)]
 for arm,checkpoint in p['checkpoints'].items():
  dest=R/f'4090-frozen-history-teacher-{arm}-sequence-fit-original-v1.json';trace=R/f'4090-frozen-history-teacher-{arm}-sequence-fit-traces-v1.npz';assert not dest.exists() and not trace.exists()
  cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==20000;m=Binary40().cuda().eval();m.load_state_dict(cp['model'],strict=True);prob=np.empty((len(rows),282),np.float32);records=[];groups={};start=time.monotonic()
  for off in range(0,len(rows),128):
   batch=rows[off:off+128];x=np.zeros((len(batch),846,40),np.float32);weights=[]
   for j,r in enumerate(batch):
    raw=sampler.binaries[r['feature_binary']][r['offset']:r['offset']+r['frames']];x[j,:r['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);_,pw,nw=masks(r['frames'],r['source']['label'],r['lead_samples'],r['container_end_samples']);weights.append(pw)
   with torch.inference_mode():
    z,_=m(torch.from_numpy(x).permute(0,2,1).cuda(),m.initial_state(len(batch),'cuda'));v=torch.sigmoid(z[:,0]).cpu().numpy();prob[off:off+len(batch)]=v
   for j,r in enumerate(batch):
    pw=weights[j];cross=crossing(v[j,:len(pw)],pw>0,False);events=float_events(v[j]);source=r['source'];key=source['engine']+'/'+source['text'];g=groups.setdefault(key,dict(rows=0,positive_crossing_rows=0,valid_event_rows=0));g['rows']+=1;g['positive_crossing_rows']+=bool(cross);g['valid_event_rows']+=any(e<=len(pw) for e in events)
    records.append(dict(index=off+j,order=r['order'],source_id=source['id'],label=source['label'],frames=r['frames'],lead_samples=r['lead_samples'],aligned_end_samples=r['container_end_samples'],positive_crossings=cross,events=events,valid_calls=len(pw)))
   if off%4096==0:print(json.dumps(dict(arm=arm,rows=off+len(batch))),flush=True)
  np.savez_compressed(trace,probability=prob);dest.write_text(json.dumps(dict(arm=arm,checkpoint_sha256=sha(checkpoint),plan_sha256=sha(P),groups=groups,records=records,trace=dict(path=str(trace),sha256=sha(trace)),optimizer_updates=0,elapsed_seconds=time.monotonic()-start,policy=p['policy']),indent=2)+'\n');print(json.dumps(dict(arm=arm,completed=True,groups=groups)),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
