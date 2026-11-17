"""One prospectively fixed parent/direct-continuation parameter midpoint."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-midpoint-plan-v1.json';B=Path('.local/wakeword/binary40-midpoint-v1')
PARENTS=[Path('.local/wakeword/binary40-training-v1/final.pt'),Path('.local/wakeword/frozen-history-teacher-training-v1/teacher/final.pt')]
EXPECTED=['f99e553b87aa18450f084eaaf9f261dcbd930c8f7d93bb2317d9dc93cbad8877','f19388bd9ce8ead14ad6218cb03f825c1816dbedff87b7d21af16837aa5f8845']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 for p,h in zip(PARENTS,EXPECTED):assert sha(p)==h
 files=[*PARENTS,Path(__file__),Path(__file__).with_name('binary40_midpoint_hypothesis_v1.md'),Path('research/wakeword_p4/training/causal_binary40_v1.py')]
 with LOCK.open('x') as f:json.dump(dict(schema='binary40-midpoint-plan-v1',alpha=.5,files={str(p):sha(p) for p in files},policy='Fixed all-tensor midpoint, no optimizer/alpha/parent/checkpoint selection. Follow complete predeclared export and8cohort evaluation, no automatic promotion.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 parents=[torch.load(p,map_location='cpu',weights_only=False) for p in PARENTS];a,b=[v['model'] for v in parents];native=Binary40();assert list(a)==list(b)==list(native.state_dict())
 assert all(c['step']==20000 for c in parents);averaged={};maxerror=0.
 for k,v in a.items():
  w=b[k];assert v.shape==w.shape==native.state_dict()[k].shape and v.dtype==w.dtype==torch.float32 and torch.isfinite(v).all() and torch.isfinite(w).all()
  averaged[k]=v*.5+w*.5;reference=((v.double()+w.double())/2).float();assert torch.equal(averaged[k],reference),k;maxerror=max(maxerror,float((averaged[k].double()-(v.double()+w.double())/2).abs().max()))
 native.load_state_dict(averaged,strict=True);B.mkdir(exist_ok=False);dest=B/'final.pt'
 torch.save(dict(model=averaged,step=20000,arm='midpoint',derivation='Fixed .5 midpoint of two completed step20000 parent checkpoints; not new optimizer updates',optimizer_updates=0,parents=[dict(path=str(p),sha256=h) for p,h in zip(PARENTS,EXPECTED)],alpha=.5,plan_sha256=sha(LOCK)),dest)
 saved=torch.load(dest,map_location='cpu',weights_only=False);assert all(torch.equal(saved['model'][k],v) for k,v in averaged.items())
 out=dict(schema='binary40-midpoint-construction-original-v1',plan_sha256=sha(LOCK),checkpoint=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),parents=saved['parents'],alpha=.5,optimizer_updates=0,tensors=len(averaged),parameters=sum(v.numel() for v in averaged.values()),float64_single_round_reference_exact=True,max_float64_rounding_error=maxerror,recognition_scored=False,limits='Tensor construction and arithmetic only; no performance, export or P4 claim.')
 (R/'4090-binary40-midpoint-construction-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
