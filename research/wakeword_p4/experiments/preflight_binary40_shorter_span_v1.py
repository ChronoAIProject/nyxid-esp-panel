"""Training-metadata and actual CUDA matched-dilation functional preflight."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-binary40-shorter-span-plan-v1.json';META=R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json';BROAD=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def lock():
 files=[Path(__file__),Path(__file__).with_name('binary40_shorter_span_proposal_v1.md'),Path('research/wakeword_p4/training/causal_binary40_span_v1.py'),Path('research/wakeword_p4/training/causal_binary40_v1.py'),Path('research/wakeword_p4/training/gpu_python.py'),Path(__file__).with_name('causal_source_balanced_batch_v1.py'),Path(__file__).with_name('causal_sequence_supervision_v1.py'),META,BROAD,R/'4090-causal-binary40-original-batch-v1.json']
 # Reuse the authenticated complete original training closure, including caches.
 old=json.loads((R/'4090-binary40-training-fit-plan-v1.json').read_text());bound=dict(old['files'])
 for f in files:bound[str(f)]=sha(f)
 with PLAN.open('x') as f:json.dump(dict(schema='binary40-shorter-span-plan-v1',files=bound,control=[1,2,4,8,10],treatment=[1,2,4,5,7],scope='Complete training-only4320native inventory/all31644broadbare metadata. Fixed2.34s treatment selected after exploratory training-reference coverage, before any modeloutputs. Original reference control exactly checked; matched same-seed tensors, independent RFdependency enumeration, CUDAfull/chunk/future/outsideRF invariance and unchanged original128samplerbatch/finitegradients/weightimmutability. No optimizer, dev/acceptance input, trained-weight swap, INT8/P4 or recognition claim. Fresh formal pairedtraining not launched.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 import torch
 from causal_source_balanced_batch_v1 import Sampler
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 from causal_binary40_span_v1 import Binary40Span,CONTROL,TREATMENT
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 inventory=json.loads(META.read_text())['inventory'];assert len(inventory)==4320 and all(r['source']['role']=='training' for r in inventory);a=[dict(id=r['source']['id'],span_samples=r['reference_samples']+4320,engine=r['source']['engine']) for r in inventory if r['source']['label']==1];assert len(a)==864
 broad=[r for r in json.loads(BROAD.read_text()) if r['source_pair_key']['carrier']=='bare' and r['text'] in ['Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.']];assert len(broad)==31644 and all(r['split']=='training' for r in broad);b=[dict(source_id=r['source_id'],variant=r['variant'],engine=r['engine'],span_samples=int(r['generated_container_end_sample']-(r['lead_frame']+2)*160+4320)) for r in broad if r['label']==1];assert len(b)==6444
 assert max(r['span_samples'] for r in a)==31070 and max(r['span_samples'] for r in b)==36000
 def dep(k,ds):
  def stem(j):return set(range(3*j-2,3*j+1))
  indices={k,k-1}
  for d in reversed(ds):indices={j-d*t for j in indices for t in range(5)}
  return set().union(*(stem(j) for j in indices))
 torch.manual_seed(726);original=Binary40().cuda().eval();torch.manual_seed(726);control=Binary40Span(CONTROL).cuda().eval();torch.manual_seed(726);treatment=Binary40Span(TREATMENT).cuda().eval()
 assert all(torch.equal(original.state_dict()[k],control.state_dict()[k]) and torch.equal(control.state_dict()[k],treatment.state_dict()[k]) for k in original.state_dict());before={k:digest(v.detach().cpu().numpy()) for k,v in original.state_dict().items()};x=torch.rand(2,40,846,device='cuda')*26;reports={}
 with torch.no_grad():
  y,_=original(x,original.initial_state(2,'cuda'));cy,_=control(x,control.initial_state(2,'cuda'));assert torch.equal(y,cy)
  for name,m in [('control',control),('treatment',treatment)]:
   g=m.geometry();dependency=dep(250,m.dilations);assert len(dependency)==g['receptive_field_frames'] and max(dependency)-min(dependency)+1==len(dependency);full,final=m(x,m.initial_state(2,'cuda'));chunk_checks=[]
   for size in [3,6,37,128,846]:
    state=m.initial_state(2,'cuda');pending=x[:,:,:0];ys=[]
    for start in range(0,846,size):
     pending=torch.cat((pending,x[:,:,start:start+size]),2);n=pending.shape[-1]//3*3
     if n:z,state=m(pending[:,:,:n],state);ys.append(z);pending=pending[:,:,n:]
    actual=torch.cat(ys,2);error=float((actual-full).abs().max());assert torch.allclose(actual,full,atol=1e-4,rtol=1e-5) and all(torch.allclose(s,t,atol=1e-4,rtol=1e-5) for s,t in zip(state,final));assert pending.shape[-1]==0;chunk_checks.append(dict(chunk_frames=size,max_abs_logit_error=error))
   changed=x.clone();changed[:,:,423:]=torch.rand_like(changed[:,:,423:])*26;future,_=m(changed,m.initial_state(2,'cuda'));assert torch.equal(future[:,:,:141],full[:,:,:141]);old=x.clone();boundary=min(dep(250,m.dilations));old[:,:,:boundary]=torch.rand_like(old[:,:,:boundary])*26;past,_=m(old,m.initial_state(2,'cuda'));assert torch.equal(past[:,:,250:],full[:,:,250:]);reports[name]=dict(geometry=g,dependency_min_frame=min(dependency),dependency_max_frame=max(dependency),dependency_frame_count=len(dependency),chunk_checks=chunk_checks,future_prior141_exact=True,outside_rf_from_call251_exact=True)
 (batch,pw,nw),descriptors=Sampler(726).batch();reference=json.loads((R/'4090-causal-binary40-original-batch-v1.json').read_text());hashes=dict(features=digest(batch),positive_weights=digest(pw),negative_weights=digest(nw));assert hashes==reference['batch_hashes'] and descriptors==reference['descriptors']
 for name,m in [('control',control),('treatment',treatment)]:
  z,_=m(torch.from_numpy(batch).permute(0,2,1).cuda(),m.initial_state(128,'cuda'));logits=z[:,0];loss=(torch.nn.functional.softplus(-logits)*torch.from_numpy(pw).cuda()+20*torch.nn.functional.softplus(logits)*torch.from_numpy(nw).cuda()).sum(1).mean();loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());assert before=={k:digest(v.detach().cpu().numpy()) for k,v in m.state_dict().items()};reports[name]['original_batch_loss']=float(loss.detach());reports[name]['all_gradients_finite_weights_unchanged']=True
 out=dict(schema='binary40-shorter-span-preflight-original-v1',plan_sha256=sha(PLAN),all_passed=True,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,inventory_sources=4320,broad_bare_rows=31644,coverage=dict(native_positive_count=len(a),native_max_reference_span_samples=max(r['span_samples'] for r in a),broad_positive_count=len(b),broad_max_reference_span_samples=max(r['span_samples'] for r in b),all_native_and_broad_positive_reference_windows_within234_frames=all(r['span_samples']<=234*160 for r in a+b),native_positive_records=a,broad_positive_records=b),same_seed_all_parameter_tensors_exact=True,original_control_logits_exact=True,models=reports,batch_hashes=hashes,optimizer_updates=0,limits=p['scope'])
 with (R/'4090-binary40-shorter-span-preflight-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['coverage','models']}),flush=True);print(json.dumps(dict(coverage={k:v for k,v in out['coverage'].items() if not k.endswith('records')},models=reports)),flush=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
