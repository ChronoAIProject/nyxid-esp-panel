"""Fixed fresh Binary40 training, original C161 sampler/BCE, final-only evaluation."""
import ast,gzip,hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-capacity-pair-plan-v1.json'
ARM=sys.argv[2] if len(sys.argv)>2 else None
B=Path('.local/wakeword/binary40-capacity-pair-v1')/str(ARM)
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
    base=Path('research/wakeword_p4/results/4090-binary40-training-plan-v1.json')
    p=json.loads(base.read_text())
    for f in [base,Path(__file__),Path('research/wakeword_p4/training/causal_binary40_wide_v1.py'),Path('.local/wakeword/binary40-training-v1/final.pt'),R/'4090-binary40-wide-initializer-export-original-v1.json',R/'4090-binary40-widen-functional-v1.json']:
        p['files'][str(f)]=sha(f)
    p.update(schema='binary40-capacity-pair-plan-v1',arms=['control64','wide128'],parent_step=20000,
       policy='Single paired continuation: original final20000 source checkpoint in both arms, control64 unchanged and wide128 fixed function-preserving split seed727/0.01RMS. Each arm original Sampler726 from firstbatch, fresh Adam1e-3, original masks/BCE,20000additional steps/final-only. Same batch schedule must be verified from logs. Width and split parameterization are combined treatment. No seed/width/LR/threshold/calibration/checkpoint rescue, dev audio or acceptance. CPU2/interop1, RTX4090 required,100GiB reserve. Separate trained final export/INT8 complete consumed diagnostics required; feasibility initializer is not final model.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert metadata.version(name)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
    assert ARM in p['arms']
    disk();B.mkdir(parents=True,exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (B/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        import torch
        from causal_source_balanced_batch_v1 import Sampler,digest
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_binary40_v1 import Binary40
        from causal_binary40_wide_v1 import widen_from
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==p['parent_step']
        model=Binary40().cuda();model.load_state_dict(cp['model'],strict=True)
        assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
        if ARM=='wide128':model=widen_from(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'],betas=tuple(p['optimizer']['betas']),eps=p['optimizer']['eps'])
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
        result=dict(schema='binary40-capacity-arm-original-v1',arm=ARM,parent_checkpoint_sha256=sha('.local/wakeword/binary40-training-v1/final.pt'),launch=launch,plan_sha256=sha(LOCK),steps=p['steps'],elapsed_seconds=time.monotonic()-start,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,progress=progress,
                    files={f.name:sha(f) for f in B.iterdir() if f.is_file()},policy=p['policy'],recognition_scored=False)
        with (R/f'4090-binary40-capacity-{ARM}-training-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(completed=True,steps=p['steps'])),flush=True)
    except Exception:
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
