"""Fixed SAM optimizer hypothesis; original data and deployment graph unchanged."""
import ast,gzip,hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-sam-training-plan-v1.json'
ARM='sam'
B=Path('.local/wakeword/binary40-sam-training-v1')/str(ARM)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sources(path,seen):
    path=Path(path)
    if path in seen:return
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text())):
        names=[]
        if isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
        elif isinstance(node,ast.Import):names=[n.name for n in node.names]
        for name in names:
            for root in [Path('research/wakeword_p4/experiments'),Path('research/wakeword_p4/training')]:
                candidate=root/(name.replace('.','/')+'.py')
                if candidate.exists():sources(candidate,seen)
def plan():
    base=R/'4090-binary40-additive-carrier-training-plan-v1.json'
    p=json.loads(base.read_text());files=dict(p['files']);paths=set()
    sources(Path(__file__),paths)
    paths.add(base);paths.add(Path(__file__).with_name('binary40_sam_protocol_v1.md'))
    preflight=R/'4090-binary40-sam-preflight-v1.json'
    assert json.loads(preflight.read_text())['all_passed']
    paths.add(preflight);paths.add(Path(__file__).with_name('preflight_binary40_sam_v1.py'))
    for f in paths:files[str(f)]=sha(f)
    p.update(schema='binary40-sam-training-plan-v1',files=files,arms=['sam'],rho=.05,
             policy='Original128 unchanged; original seed726 fresh Adam1e-3, SAM rho.05 L2 global gradient, no adaptive scaling, two forwards on identical batch;20000finalonly. CPU2 interop1 GPU4090 reserve100GiB. No dev/heldout access or parameter/checkpoint rescue. C161 unchanged.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert metadata.version(name)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
    assert ARM in p['arms'];disk();B.mkdir(parents=True,exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (B/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        import torch
        from causal_source_balanced_batch_v1 import Sampler as OriginalSampler,digest
        from sam_step_v1 import sam_gradients
        from paired_carrier_batch_audit_v1 import audit
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_binary40_span_v1 import Binary40Span,CONTROL
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        model=Binary40Span(CONTROL).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'],betas=tuple(p['optimizer']['betas']),eps=p['optimizer']['eps'])
        parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
        assert model.state_dict().keys()==parent.keys() and all(torch.equal(v.cpu(),parent[k]) for k,v in model.state_dict().items())
        sampler=OriginalSampler(p['seed']);reference=OriginalSampler(p['seed']);start=time.monotonic();progress=[]
        def save(name,step):
            tmp=B/(name+'.tmp');torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),geometry=model.geometry(),arm=ARM,plan_sha256=sha(LOCK)),tmp);tmp.replace(B/(name+'.pt'))
        save('initial',0)
        with gzip.open(B/'batches.jsonl.gz','wt') as log:
            for step in range(1,p['steps']+1):
                (x,pw,nw),descriptors=sampler.batch()
                baseline,baseline_descriptors=reference.batch()
                size=128
                assert len(descriptors)==size and descriptors[:128]==baseline_descriptors
                assert all(np.array_equal(a[:128],b) for a,b in zip((x,pw,nw),baseline,strict=True))
                batch_audit=dict(original128_exact=True,appended_rows=0)
                assert x.shape==(size,846,40) and pw.shape==nw.shape==(size,282)
                tx=torch.from_numpy(x).permute(0,2,1).cuda();tp=torch.from_numpy(pw).cuda();tn=torch.from_numpy(nw).cuda()
                def closure():
                    z,_=model(tx,model.initial_state(size,'cuda'));z=z[:,0,:]
                    return (torch.nn.functional.softplus(-z)*tp+20*torch.nn.functional.softplus(z)*tn).sum()/128
                loss,perturbed_loss,gradient_norm=sam_gradients(model,optimizer,closure,p['rho'])
                optimizer.step();assert all(torch.isfinite(v).all() for v in model.parameters())
                log.write(json.dumps(dict(step=step,loss=float(loss.detach()),perturbed_loss=perturbed_loss,gradient_norm=gradient_norm,features_sha256=digest(x),positive_sha256=digest(pw),negative_sha256=digest(nw),descriptors=descriptors,batch_audit=batch_audit))+'\n')
                if step==1 or step%p['progress_every']==0 or step==p['steps']:
                    row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),all_gradients_and_parameters_finite=True)
                    progress.append(row);log.flush();(B/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
                if step%p['disk_check_every']==0:save('latest',step);disk()
        save('final',p['steps'])
        result=dict(schema='binary40-sam-training-original-v1',arm=ARM,geometry=model.geometry(),launch=launch,plan_sha256=sha(LOCK),steps=p['steps'],elapsed_seconds=time.monotonic()-start,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,progress=progress,
                    files={f.name:sha(f) for f in B.iterdir() if f.is_file()},policy=p['policy'],recognition_scored=False)
        with (R/f'4090-binary40-sam-{ARM}-training-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(completed=True,steps=p['steps'])),flush=True)
    except Exception:
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
