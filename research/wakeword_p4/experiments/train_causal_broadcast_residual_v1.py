"""Fixed fresh causal broadcast residual training, original C161 sampler/BCE, final-only evaluation."""
import ast,gzip,hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-broadcast-residual-training-plan-v1.json'
B=Path('.local/wakeword/broadcast-residual-training-v1')
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
    parent=R/'4090-causal-source-balanced-plan-v1.json';base=json.loads(parent.read_text())
    paths=set()
    for f in [Path(__file__),Path(__file__).with_name('causal_source_balanced_batch_v1.py'),Path('research/wakeword_p4/training/causal_broadcast_residual_v1.py')]:sources(f,paths)
    paths.update([Path(__file__).with_name('preflight_broadcast_residual_training_v1.py'),R/'4090-broadcast-residual-training-preflight-v1.json',R/'4090-broadcast-residual-feasibility-untrained-export-original-v1.json',Path(__file__).with_name('causal_broadcast_residual_proposal_v1.md'),parent,R/'4090-causal-binary40-original-batch-v1.json',R/'4090-binary40-feasibility-original-v1.json',Path('research/wakeword_p4/tools/check_disk_budget.py')])
    files=dict(base['files']);files.update({str(f):sha(f) for f in paths})
    p=dict(schema='broadcast-residual-training-plan-v1',files=files,versions={n:metadata.version(n) for n in ['torch','numpy']},seed=726,
           steps=20000,batch_size=128,padded_frames=846,optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8),
           progress_every=200,disk_check_every=1000,expected_new_gib=1,
           hypothesis='Frequency-retained broadcast residual full network;7457parameters/3312state elements. Combined change of width, spectral/temporal routing and pooling, no BN/SSN/Swish. Not BC-ResNet reproduction, pure-locality attribution or equal-compute control. Original sampler/masks/Adam fixed.',
           policy='Fresh seed726, originalSampler726 full128rows unchanged; original region-normalized positive+20negative BCE, Adam1e-3,20000finalonly. No appended pairs, dev/acceptance audio, augmentation/threshold/checkpoint/seed rescue. CPU2interop1 GPU4090 reserve100GiB. Exhaustive original batch ledger audit, training-only broad support and complete8cohort/3context comparison required after final single original calibration INT8 export. Historical originalBinary40 reference is not simultaneous equal-compute control; C161 remains baseline. P4 resources/kernels/firstcommand/stability unproven.')
    pre=json.loads((R/'4090-broadcast-residual-training-preflight-v1.json').read_text());assert pre['all_passed'] and pre['optimizer_updates']==0
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),source_files=len(paths),dependency_files=len(files))))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert metadata.version(name)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
    disk();B.mkdir(exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (B/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        import torch
        from causal_source_balanced_batch_v1 import Sampler,digest
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_broadcast_residual_v1 import CausalBroadcastResidual
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        model=CausalBroadcastResidual().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'],betas=tuple(p['optimizer']['betas']),eps=p['optimizer']['eps'])
        sampler=Sampler(p['seed']);start=time.monotonic();progress=[]
        def save(name,step):
            tmp=B/(name+'.tmp');torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),plan_sha256=sha(LOCK)),tmp);tmp.replace(B/(name+'.pt'))
        save('initial',0)
        with gzip.open(B/'batches.jsonl.gz','wt') as log:
            for step in range(1,p['steps']+1):
                (x,pw,nw),descriptors=sampler.batch()
                assert x.shape==(128,846,40) and pw.shape==nw.shape==(128,282)
                optimizer.zero_grad(set_to_none=True)
                z,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(128,'cuda'));z=z[:,0,:]
                loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(pw).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(nw).cuda()).sum(1).mean()
                assert torch.isfinite(loss);loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
                optimizer.step();assert all(torch.isfinite(v).all() for v in model.parameters())
                log.write(json.dumps(dict(step=step,loss=float(loss.detach()),features_sha256=digest(x),positive_sha256=digest(pw),negative_sha256=digest(nw),descriptors=descriptors))+'\n')
                if step==1 or step%p['progress_every']==0 or step==p['steps']:
                    row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),all_gradients_and_parameters_finite=True)
                    progress.append(row);log.flush();(B/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
                if step%p['disk_check_every']==0:save('latest',step);disk()
        save('final',p['steps'])
        result=dict(schema='broadcast-residual-training-original-v1',launch=launch,plan_sha256=sha(LOCK),steps=p['steps'],elapsed_seconds=time.monotonic()-start,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,progress=progress,
                    files={f.name:sha(f) for f in B.iterdir() if f.is_file()},policy=p['policy'],recognition_scored=False)
        with (R/'4090-broadcast-residual-training-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(completed=True,steps=p['steps'])),flush=True)
    except Exception:
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
