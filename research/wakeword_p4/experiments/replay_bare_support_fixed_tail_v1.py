"""All23 training bare sources plus one fixed8000sample tail, diagnostic only."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-bare-support-fixed-tail-plan-v1.json';OUT=Path('.local/wakeword/bare-support-fixed-tail-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
    admission=R/'4090-chatterbox-bare-training-admission-v1.json';export=R/'4090-context-bare-support-int8-original-v1.json';e=json.loads(export.read_text());model=Path(e['model']['path']);assert sha(model)==e['model']['sha256']
    rows=json.loads(admission.read_text())['admitted'];assert len(rows)==23
    paths=[Path(__file__),admission,export,model,Path('research/wakeword_p4/training/features.py'),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py')]
    paths += [Path(r['wav']['path']) for r in rows]
    p=dict(files={str(p):sha(p) for p in paths},model=str(model),rows=rows,tail_samples=8000,policy='All23consumed provisionaltrainingbare positives, append exactly8000zeros after unchanged fullPCM. Fixed .91796875/5/67/half-rearm/81600continuousprime/native480,CPU1thread deployment simulation. No threshold/seed/checkpoint/calibration tuning. Tail-only diagnostic motivated after saved end-of-wave observations, not independent validation; never overwrite original20/23 or claim generalization/P4/phonetic truth. Preserve delayed events separately.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    from collect_multispeaker_formal_v1 import prepare,predict
    from uint8_event_policy_v1 import event_steps
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
    OUT.mkdir(exist_ok=False);records=[]
    for row in p['rows']:
        pcm,rate=sf.read(row['wav']['path'],dtype='int16');assert rate==16000 and pcm.ndim==1
        wave=np.concatenate([np.zeros(81600,np.int16),pcm,np.zeros(8000,np.int16)])
        assert np.array_equal(wave[81600:81600+len(pcm)],pcm)
        values,support,_=prepare(features(wave),len(wave));clock=np.minimum(((support+479)//480)*480,len(wave))-81600
        def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
        _,codes,q=predict(values,p['model'],runtime)
        events=[dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)]
        trace=OUT/f'{row["index"]:03d}.npz';np.savez_compressed(trace,uint8=codes,relative_block_return=clock,support=support)
        records.append(dict(index=row['index'],source=row,original_samples=len(pcm),events=events,real_events=[e for e in events if e['relative_sample']>0],after_original_end_events=[e for e in events if e['relative_sample']>len(pcm)],trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),quantization=q))
    result=dict(plan_sha256=sha(PLAN),records=records,policy=p['policy'])
    with (OUT/'result.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(clips=23,hit=sum(bool(r['real_events']) for r in records),after_end=sum(bool(r['after_original_end_events']) for r in records))))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
