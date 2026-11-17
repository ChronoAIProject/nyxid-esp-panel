"""All4320 teacher contradiction census plus first50 no-update actual gradients."""
import gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-frozen-history-teacher-network-plan-v1.json';CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def plan():
 cache=json.loads((R/'4090-frozen-history-teacher-cache-original-v1.json').read_text());old=json.loads((R/'4090-binary40-training-plan-v1.json').read_text());files=dict(old['files'])
 paths=[Path(__file__),Path(__file__).with_name('frozen_history_teacher_pair_v1.py'),CP,Path('.local/wakeword/binary40-training-v1/batches.jsonl.gz')]+[R/n for n in ['4090-frozen-history-teacher-cache-original-v1.json','4090-frozen-history-teacher-metadata-math-preflight-v1.json','4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz']]+[Path(f['path']) for f in cache['files'].values()]
 files.update({str(p):sha(p) for p in paths});PLAN.write_text(json.dumps(dict(files=files,policy='All4320native-source quiet/speech frozen teacher probabilities; original positive mask-contained five-probability crossings, negative post-target crossings/mean target contradictions. First50 original main128batches authenticated against saved original hashes plus paired schedule4perstep; supervised main+pair BCE and independent fixed teacher KL gradients, no optimizer. Exact repeat paired tensors, teacher gradient disconnected and teacher/student state unchanged. Save all probabilities and component gradients; no tuning coefficient or source selection. Teacher contradiction does not invalidate metadata automatically; report all labels/voices.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from frozen_history_teacher_pair_v1 import Pairs,tensor,bce,pair_components
 from causal_source_balanced_batch_v1 import Sampler,digest
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load(CP,map_location='cpu',weights_only=False);assert cp['step']==20000
 student=Binary40().cuda().eval();student.load_state_dict(cp['model'],strict=True);teacher=Binary40().cuda().eval();teacher.load_state_dict(cp['model'],strict=True);teacher.requires_grad_(False)
 before={k:v.detach().clone() for k,v in student.state_dict().items()};pairs=Pairs();arrays={};census=[];start=time.monotonic()
 with torch.no_grad():
  for begin in range(0,4320,16):
   ids=list(range(begin,min(begin+16,4320)));x,pw,nw,tw=pairs.batch(ids);z,_=teacher(tensor(x).permute(0,2,1),teacher.initial_state(len(x),'cuda'));probs=torch.sigmoid(z[:,0]).cpu().numpy()
   for j,i in enumerate(ids):
    row=pairs.meta[i];arms={}
    for view,h in enumerate(['silence','speech']):
     v=probs[2*j+view,:row['frames']//3];arrays[f'p_{i}_{h}']=v;mask=pw[2*j+view,:len(v)]>0 if row['source']['label'] else tw[j,:len(v)]>0;five=np.convolve(v.astype(np.float64),np.ones(5)/5,'valid');valid=np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5
     assert valid.any();arms[h]=dict(crossing=bool(np.any(five[valid]>=.91796875)),max_five=float(five[valid].max()),mean_probability=float(v[mask].mean()),mean_contradicts_label=bool((v[mask].mean()>=.5)!=bool(row['source']['label'])))
    census.append(dict(index=i,source_id=row['source']['id'],label=row['source']['label'],engine=row['source']['engine'],speaker_id=row['source']['speaker_id'],arms=arms))
   if begin%512==0:print(json.dumps(dict(census_rows=len(census))),flush=True)
 sampler=Sampler(726);probe=[];original=gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt');params=tuple(student.parameters())
 for step in range(1,51):
  (x,pw,nw),desc=sampler.batch();old=json.loads(next(original));assert old['step']==step and old['descriptors']==desc
  for key,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(a)==old[key]
  ids=pairs.schedule[step-1]['inventory_indices'];batch=pairs.batch(ids);again=pairs.batch(ids);assert all(np.array_equal(a,b) for a,b in zip(batch,again))
  logits,_=student(tensor(x).permute(0,2,1),student.initial_state(len(x),'cuda'));main=bce(logits[:,0],tensor(pw),tensor(nw));pair,kl=pair_components(student,teacher,batch);sup=main+pair
  gs=torch.autograd.grad(sup,params,retain_graph=True);gd=torch.autograd.grad(kl,params)
  s=np.concatenate([g.detach().cpu().numpy().ravel() for g in gs]);d=np.concatenate([g.detach().cpu().numpy().ravel() for g in gd]);assert np.isfinite(s).all() and np.isfinite(d).all();arrays[f'gS_{step}']=s;arrays[f'gD_{step}']=d
  sn=float(np.linalg.norm(s.astype(np.float64)));dn=float(np.linalg.norm(d.astype(np.float64)));dot=float(np.dot(s.astype(np.float64),d.astype(np.float64)))
  probe.append(dict(step=step,inventory_indices=ids,pair_tensor_hashes=[digest(a) for a in batch],main_bce=float(main.detach()),paired_bce=float(pair.detach()),kl=float(kl.detach()),supervised_norm=sn,teacher_kl_norm=dn,dot=dot,cosine=dot/(sn*dn) if sn*dn else None));assert all(t.grad is None for t in teacher.parameters())
 original.close();assert all(torch.equal(before[k],v) for k,v in student.state_dict().items());assert all(torch.equal(before[k],v) for k,v in teacher.state_dict().items())
 trace=R/'4090-frozen-history-teacher-network-traces-v1.npz';np.savez_compressed(trace,**arrays)
 out=dict(plan_sha256=sha(PLAN),census=census,probes=probe,optimizer_updates=0,teacher_student_unchanged=True,teacher_disconnected=True,trace=dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size),elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),policy=p['policy'])
 (R/'4090-frozen-history-teacher-network-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(census=len(census),probes=len(probe),elapsed_seconds=out['elapsed_seconds'])))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
