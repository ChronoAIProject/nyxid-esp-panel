"""Final-weight all4320 physical history fit, no optimization/evaluation sources."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ARM=sys.argv[2];assert ARM in ['control','teacher'];T=Path('.local/wakeword/frozen-history-teacher-training-v1')/ARM;PLAN=R/f'4090-frozen-history-teacher-{ARM}-physical-fit-plan-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def plan():
 old=json.loads((R/'4090-frozen-history-teacher-network-plan-v1.json').read_text());files=dict(old['files'])
 for p in [Path(__file__),T/'final.pt',R/'4090-frozen-history-teacher-network-original-v1.json',R/'4090-frozen-history-teacher-training-review-v1.json',R/f'4090-frozen-history-teacher-{ARM}-training-original-v1.json']:files[str(p)]=sha(p)
 PLAN.write_text(json.dumps(dict(arm=ARM,files=files,policy='Final20000only, all4320original physical quiet/history training pairs; complete fixed cached features, no mining/optimizer/dev/acceptance access. Same original teacher census positive-mask-contained and negative posttarget five-mean metrics, compare every source identity and row gains/losses. GPUfloat training-fit only, not INT8/generalization/P4 or FAR. No threshold/calibration/checkpoint rescue.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 assert json.loads((R/'4090-frozen-history-teacher-training-review-v1.json').read_text())['all_passed']
 import torch
 from frozen_history_teacher_pair_v1 import Pairs,tensor
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load(T/'final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000 and cp['arm']==ARM;m=Binary40().cuda().eval();m.load_state_dict(cp['model'],strict=True)
 pairs=Pairs();baseline=json.loads((R/'4090-frozen-history-teacher-network-original-v1.json').read_text())['census'];arrays={};records=[];summary={};start=time.monotonic()
 with torch.no_grad():
  for begin in range(0,4320,16):
   ids=list(range(begin,min(begin+16,4320)));x,pw,nw,tw=pairs.batch(ids);z,_=m(tensor(x).permute(0,2,1),m.initial_state(len(x),'cuda'));probs=torch.sigmoid(z[:,0]).cpu().numpy()
   for j,i in enumerate(ids):
    row=pairs.meta[i];old=baseline[i];assert old['source_id']==row['source']['id'];arms={};label=row['source']['label'];g=summary.setdefault(str(label),dict(rows=0,silence_crossings=0,speech_crossings=0));g['rows']+=1
    for view,h in enumerate(['silence','speech']):
     v=probs[2*j+view,:row['frames']//3];arrays[f'p_{i}_{h}']=v;mask=pw[2*j+view,:len(v)]>0 if label else tw[j,:len(v)]>0;five=np.convolve(v.astype(np.float64),np.ones(5)/5,'valid');valid=np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5;hit=bool(np.any(five[valid]>=.91796875));g[h+'_crossings']+=hit
     arms[h]=dict(crossing=hit,max_five=float(five[valid].max()),original_teacher_crossing=old['arms'][h]['crossing'])
    records.append(dict(index=i,source_id=row['source']['id'],label=label,arms=arms))
 trace=R/f'4090-frozen-history-teacher-{ARM}-physical-fit-traces-v1.npz';np.savez_compressed(trace,**arrays);out=dict(arm=ARM,plan_sha256=sha(PLAN),checkpoint_sha256=sha(T/'final.pt'),records=records,summary=summary,trace=dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size),elapsed_seconds=time.monotonic()-start,optimizer_updates=0,policy=p['policy']);(R/f'4090-frozen-history-teacher-{ARM}-physical-fit-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
