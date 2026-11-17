"""Fixed matched fresh306/234span training; identical original inputs/BCE."""
import ast,gzip,hashlib,importlib.metadata as metadata,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-shorter-span-training-plan-v1.json'
ARM=sys.argv[2] if len(sys.argv)>2 else None
B=Path('.local/wakeword/binary40-shorter-span-training-v1')/str(ARM)
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
    base=R/'4090-binary40-training-plan-v1.json';p=json.loads(base.read_text());files=dict(p['files']);paths=set()
    for script in ['train_binary40_shorter_span_pair_v1.py','review_binary40_shorter_span_training_v1.py','export_binary40_shorter_span_final_v1.py','replay_binary40_shorter_span_v1.py','score_binary40_shorter_span_v1.py','run_binary40_shorter_span_context_v1.py','continue_binary40_shorter_span_pair_v1.py','review_binary40_shorter_span_consumed_v1.py']:
        sources(Path(__file__).with_name(script),paths)
    paths.update([base,Path('research/wakeword_p4/training/causal_binary40_span_v1.py'),Path(__file__).with_name('build_binary40_span_keras_v1.py'),R/'4090-binary40-shorter-span-feasibility-original-v1.json',R/'4090-binary40-shorter-span-feasibility-review-v1.json',R/'4090-binary40-shorter-span-feasibility-execution-v1.json',Path('.local/wakeword/binary40-training-v1/initial.pt'),Path('.local/wakeword/binary40-training-v1/batches.jsonl.gz'),Path(__file__).with_name('binary40_shorter_span_formal_protocol_v1.md')])
    for name in ['4090-c161-binary40-parallel-runtime-plan-v1.json','4090-parallel-context-streams-plan-v1.json']:
        closure=R/name;files.update(json.loads(closure.read_text())['files']);paths.add(closure)
    for name in ['replay_daily_language_pair_v1.py','replay_continuous_chatterbox_v1.py','collect_melo_development_v2.py','replay_c161_chatterbox_mac_witness_v1.py','collect_multispeaker_priming_v1.py']:
        sources(Path(__file__).with_name(name),paths)
    for f in paths:files[str(f)]=sha(f)
    assert json.loads((R/'4090-binary40-shorter-span-feasibility-review-v1.json').read_text())['all_passed']
    p.update(schema='binary40-shorter-span-training-plan-v1',files=files,arms=['control','shortspan'],dilations=dict(control=[1,2,4,8,10],shortspan=[1,2,4,5,7]),expected_new_gib=2,
        hypothesis='Single matched fresh same-width/seed/source/optimizer comparison; only lasttwo depthwise dilations change, neural RF306vs234 and cache6544vs5008. Smaller neural span may reduce stale context, but adaptive frontend history persists and slower/longer/lexical coverage may regress.',
        export_profile=dict(training_rows=list(range(16)),calibration_call_indices=list(range(0,282,40)),final_step=20000,parity_calls=64,feedback_calls=256),
        consumed_scope=dict(cohorts=['builtin','daily','continuous','primed','jenny','joe','melo','mac'],new_context_streams=['builtin','melo-US','melo-BR']),
        policy='Sequential freshcontrol306 and freshshortspan234, seed726sameinitial tensors, original Sampler726128/BCEpositive+20negative/Adam1e-3/20000updates each. Dilationsoleintervention; no pairedhistory/teacher/newloss/newdata/crops/labels/augmentation. Preserve all20000batch identities against eachother/original, actualexit receipts and finalonly checkpoints. Beforefit entire source/pipeline bound; finaltraining-state128representatives/singleINT8export each with ownmetadata, all8consumed cohorts+3newcontext streams fixed235/256/5/67/81600continuousprime. Allordinary gains/losses and commonname/background conflicts preserved, hours nonadditive. No threshold/seed/dilation/coefficient/checkpoint/calibration rescue, acceptance tuning, partialcohortpromotion or P4claim. CPU2/interop1, replayBUILTIN_REF1thread, GPU4090required, >=100GiBdiskreserve; no restartonobservationtimeout.')
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
        from causal_source_balanced_batch_v1 import Sampler,digest
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_binary40_span_v1 import Binary40Span
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        model=Binary40Span(tuple(p['dilations'][ARM])).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=p['optimizer']['lr'],betas=tuple(p['optimizer']['betas']),eps=p['optimizer']['eps'])
        sampler=Sampler(p['seed']);start=time.monotonic();progress=[]
        def save(name,step):
            tmp=B/(name+'.tmp');torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),geometry=model.geometry(),arm=ARM,plan_sha256=sha(LOCK)),tmp);tmp.replace(B/(name+'.pt'))
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
        result=dict(schema='binary40-shorter-span-training-original-v1',arm=ARM,geometry=model.geometry(),launch=launch,plan_sha256=sha(LOCK),steps=p['steps'],elapsed_seconds=time.monotonic()-start,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,progress=progress,
                    files={f.name:sha(f) for f in B.iterdir() if f.is_file()},policy=p['policy'],recognition_scored=False)
        with (R/f'4090-binary40-shorter-span-{ARM}-training-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(dict(completed=True,steps=p['steps'])),flush=True)
    except Exception:
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
