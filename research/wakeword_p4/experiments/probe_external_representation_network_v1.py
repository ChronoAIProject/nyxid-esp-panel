"""Full source/context and first50actual-gradient preflight, zero updates."""
import gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-external-representation-network-plan-v1.json';B=Path('.local/wakeword/external-representation-network-v1');CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 from train_binary40_v1 import sources
 previous=json.loads((R/'4090-frozen-history-teacher-network-plan-v1.json').read_text());files=dict(previous['files']);paths=set();sources(__file__,paths)
 paths.update([R/'4090-external-representation-target-cache-original-v1.json',R/'4090-external-representation-target-review-v1.json',Path('.local/wakeword/external-representation-target-cache-v1/targets.npz'),Path(__file__).with_name('external_representation_network_protocol_v1.md'),Path('research/wakeword_p4/training/gpu_python.py')])
 files.update({str(p):sha(p) for p in paths});assert sha(CP)=='f99e553b87aa18450f084eaaf9f261dcbd930c8f7d93bb2317d9dc93cbad8877'
 with LOCK.open('x') as f:json.dump(dict(schema='external-representation-network-plan-v1',files=files,projection_seed=726,source_count=4320,gradient_batches=50,optimizer_updates=0,policy='Full4320original physical pairs and30first-stratum two-chunk checks; first50original main128/schedule4pair gradient components, independent cosine/BCE math, exact student/projection/target immutability. No optimizer, teacher inference, development/acceptance or student efficacy/P4 claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.2'],check=True);B.mkdir(exist_ok=False)
 try:
  import torch
  from external_representation_pair_v1 import Targets,context_forward,components,tensor,bce
  from causal_source_balanced_batch_v1 import Sampler,digest
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
  from causal_binary40_v1 import Binary40
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
  cp=torch.load(CP,map_location='cpu',weights_only=False);model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True);torch.manual_seed(726);projection=torch.nn.Linear(128,768).cuda()
  before={k:v.detach().clone() for k,v in model.state_dict().items()};pbefore={k:v.detach().clone() for k,v in projection.state_dict().items()};store=Targets();target_before=store.gpu_targets.clone();arrays={};records=[];checks=[];seen=set();started=time.monotonic()
  with torch.no_grad():
   for begin in range(0,4320,16):
    ids=list(range(begin,min(begin+16,4320)));batch=store.batch(ids);(x,pw,nw,tw),target,anchors=batch;xt=tensor(x).permute(0,2,1);z,states,c=context_forward(model,xt,model.initial_state(len(x),'cuda'))
    reconstructed=torch.einsum('bct,c->bt',c,model.head.weight.reshape(-1))+model.head.bias;error=float((z[:,0]-reconstructed).abs().max());assert torch.allclose(z[:,0],reconstructed,atol=2e-4,rtol=2e-4)
    for j,i in enumerate(ids):
     anchor=int(anchors[2*j]);selected=c[2*j:2*j+2,:,anchor].cpu().numpy();arrays[f'context_{i}']=selected;arrays[f'logit_{i}']=z[2*j:2*j+2,0,anchor].cpu().numpy();m=store.pairs.meta[i];key=(m['source']['engine'],m['source']['speaker_id'])
     records.append(dict(index=i,source_id=m['source']['id'],anchor_zero_based_call=anchor,source_context_sha256=digest(selected),head_full_max_error=error))
     if key not in seen:
      seen.add(key);xx=xt[2*j:2*j+2];split=3*anchor;assert 0<split<xx.shape[2];full,fs,fc=context_forward(model,xx,model.initial_state(2,'cuda'));left,ls,lc=context_forward(model,xx[:,:,:split],model.initial_state(2,'cuda'));right,rs,rc=context_forward(model,xx[:,:,split:],ls)
      assert torch.allclose(torch.cat((left,right),2),full,atol=2e-4,rtol=2e-4) and torch.allclose(torch.cat((lc,rc),2),fc,atol=2e-4,rtol=2e-4) and all(torch.allclose(a,b,atol=2e-4,rtol=2e-4) for a,b in zip(fs,rs))
      checks.append(dict(source_id=m['source']['id'],split_feature=split,anchor_context_max_error=float((rc[:,:,0]-fc[:,:,anchor]).abs().max()),logit_max_error=float((torch.cat((left,right),2)-full).abs().max()),context_max_error=float((torch.cat((lc,rc),2)-fc).abs().max())))
    if begin%512==0:print(json.dumps(dict(source_pairs=len(records))),flush=True)
  assert len(records)==4320 and len(seen)==30
  sampler=Sampler(726);params=tuple(model.parameters());extra=tuple(projection.parameters());names=[n for n,_ in model.named_parameters()];probes=[]
  with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as original:
   for step in range(1,51):
    (x,pw,nw),desc=sampler.batch();old=json.loads(next(original));assert old['step']==step and old['descriptors']==desc
    for key,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(a)==old[key]
    schedule=store.pairs.schedule[step-1];assert schedule['step']==step;ids=schedule['inventory_indices'];batch=store.batch(ids);again=store.batch(ids);assert all(np.array_equal(a,b) for a,b in zip(batch[0],again[0])) and np.array_equal(batch[1],again[1]) and np.array_equal(batch[2],again[2])
    logits,_=model(tensor(x).permute(0,2,1),model.initial_state(128,'cuda'));main=bce(logits[:,0],tensor(pw),tensor(nw));paired,feature,z,context,q,target=components(model,projection,batch);supervised=main+paired
    actual_q_gradient=torch.autograd.grad(feature,q,retain_graph=True)[0].detach().cpu().numpy();qq=q.detach().cpu().numpy().astype(np.float64);tt=target.detach().cpu().numpy().astype(np.float64);qn=np.linalg.norm(qq,axis=1,keepdims=True);tn=np.linalg.norm(tt,axis=1,keepdims=True);assert np.all(qn>1e-12) and np.all(tn>1e-12)
    analytical=(-tt/(qn*tn)+(qq*tt).sum(1,keepdims=True)*qq/(qn**3*tn))/len(qq);math_error=float(np.abs(analytical-actual_q_gradient).max());assert np.allclose(analytical,actual_q_gradient,atol=1e-6,rtol=0)
    gs=torch.autograd.grad(supervised,params,retain_graph=True);gd=torch.autograd.grad(feature,params+extra,allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in gs)
    for name,g in zip(names,gd[:len(params)]):assert (g is None) if name.startswith('head.') else (g is not None and torch.isfinite(g).all())
    assert all(g is not None and torch.isfinite(g).all() for g in gd[len(params):])
    ss=np.concatenate([v.detach().cpu().numpy().ravel() for v in gs]);dd=np.concatenate([(g.detach().cpu().numpy() if g is not None else np.zeros(tuple(v.shape),np.float32)).ravel() for v,g in zip(params,gd[:len(params)])]);student_norm=float(np.linalg.norm(dd.astype(np.float64)));assert student_norm>0
    wgrad=gd[len(params)].detach().cpu().numpy();bgrad=gd[len(params)+1].detach().cpu().numpy();ctx=context.detach().cpu().numpy();projection_error=float(np.abs(wgrad-analytical.T@ctx.astype(np.float64)).max());assert np.allclose(wgrad,analytical.T@ctx.astype(np.float64),atol=1e-6,rtol=1e-5) and np.allclose(bgrad,analytical.sum(0),atol=1e-6,rtol=1e-5)
    if step==1:
     zz=z.detach().cpu().numpy().astype(np.float64);positive=batch[0][1].astype(np.float64);negative=batch[0][2].astype(np.float64);manual=float((np.logaddexp(0,-zz)*positive+20*np.logaddexp(0,zz)*negative).sum(1).mean());assert np.isclose(manual,float(paired.detach()),atol=1e-5,rtol=1e-5)
    sn=float(np.linalg.norm(ss.astype(np.float64)));dot=float(ss.astype(np.float64)@dd.astype(np.float64));arrays.update({f'gS_{step}':ss,f'gD_{step}':dd,f'probe_context_{step}':ctx,f'probe_q_{step}':qq,f'probe_target_{step}':tt})
    probes.append(dict(step=step,indices=ids,pair_tensor_hashes=[digest(a) for a in batch[0]],target_sha256=digest(batch[1]),anchors=batch[2].tolist(),main_bce=float(main.detach()),paired_bce=float(paired.detach()),cosine_loss=float(feature.detach()),supervised_norm=sn,feature_student_norm=student_norm,feature_projection_norm=float(np.sqrt(sum(float(g.detach().double().square().sum()) for g in gd[len(params):]))),dot=dot,cosine=dot/(sn*student_norm) if sn else None,cosine_math_max_error=math_error,projection_gradient_math_max_error=projection_error))
  assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and all(torch.equal(pbefore[k],v) for k,v in projection.state_dict().items()) and torch.equal(target_before,store.gpu_targets)
  assert all(v.grad is None for v in [*model.parameters(),*projection.parameters()]) and not model.head._forward_pre_hooks
  arrays.update(projection_weight=projection.weight.detach().cpu().numpy(),projection_bias=projection.bias.detach().cpu().numpy(),binary_head_weight=model.head.weight.detach().cpu().numpy(),binary_head_bias=model.head.bias.detach().cpu().numpy())
  trace=B/'traces.npz';np.savez_compressed(trace,**arrays);out=dict(schema='external-representation-network-original-v1',plan_sha256=sha(LOCK),records=records,two_chunk_checks=checks,probes=probes,optimizer_updates=0,teacher_inference_calls=0,student_projection_targets_unchanged=True,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),elapsed_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),policy=p['policy'])
  with (R/'4090-external-representation-network-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
  print(json.dumps(dict(completed=True,source_pairs=4320,two_chunk_checks=30,gradient_batches=50,elapsed_seconds=out['elapsed_seconds'])),flush=True)
 except Exception:
  import traceback
  (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
