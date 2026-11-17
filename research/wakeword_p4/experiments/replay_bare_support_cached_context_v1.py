"""Complete92 actual cached training histories; not physical continuous startup."""
import json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from causal_sequence_supervision_v1 import masks
R=Path('research/wakeword_p4/results');CACHE=Path('.local/wakeword/chatterbox-bare-context-cache-v1');PLAN=R/'4090-bare-support-cached-context-plan-v1.json';OUT=Path('.local/wakeword/bare-support-cached-context-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
    epath=R/'4090-context-bare-support-int8-original-v1.json';e=json.loads(epath.read_text());model=Path(e['model']['path']);assert sha(model)==e['model']['sha256']
    files=[Path(__file__),epath,model,CACHE/'records.json',CACHE/'result.json',CACHE/'speech-features.bin',CACHE/'silence-features.bin',Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py'),Path(__file__).with_name('causal_sequence_supervision_v1.py')]
    p=dict(files={str(f):sha(f) for f in files},model=str(model),policy='All46cachedpairs x2history conditions,23provisionaltrainingbare sources. Actual fullfrontend then trainingcrop geometry, original target+8000tail, fresh zero model state atcrop; not physical fullwave/81600prime equivalence. Fixed .91796875/5/67/half-rearm/native480,CPU1thread simulation. No training/threshold tuning/acceptance/selection. Compare saved isolated20/23 andtail21/23 descriptively, no retroactive rescoring or qualification.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    from collect_multispeaker_formal_v1 import prepare,predict
    from uint8_event_policy_v1 import event_steps
    def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    OUT.mkdir(exist_ok=False);records=[]
    for condition in ['speech','silence']:
        data=np.memmap(CACHE/f'{condition}-features.bin',dtype='<u2',mode='r').reshape(-1,40)
        for row in json.loads((CACHE/'records.json').read_text()):
            raw=data[row['offset']:row['offset']+row['frames']];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['feature_sha256'][condition]
            values,support,_=prepare(raw,row['frames']*160+320);_,codes,q=predict(values,p['model'],runtime)
            support,pw,nw=masks(row['frames'],1,row['lead'],row['endpoint']);clock=((support+479)//480)*480-row['lead']
            events=[dict(step=k,relative_sample=int(clock[k-1])) for k in event_steps(codes,.91796875)]
            trace=OUT/f'{row["pair"]:03d}-{condition}.npz';np.savez_compressed(trace,uint8=codes,relative_block_return=clock,support=support,positive_weights=pw,negative_weights=nw)
            sums=np.convolve(codes.astype(np.int64),np.ones(5,dtype=np.int64),'valid');target=clock[4:]>0
            records.append(dict(pair=row['pair'],index=row['source']['index'],condition=condition,gap_samples=row['gap_samples'],events=events,target_events=[e for e in events if e['relative_sample']>0],max_target_sum5=int(sums[target].max()),positive_mask_mean_code=float(codes[pw>0].mean()),positive_calls=np.flatnonzero(pw).tolist(),trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),quantization=q))
    with (OUT/'result.json').open('x') as f:f.write(json.dumps(dict(plan_sha256=sha(PLAN),records=records,policy=p['policy']),indent=2)+'\n')
    for c in ['speech','silence']:
        for gap in [8000,32000]:
            rs=[r for r in records if r['condition']==c and r['gap_samples']==gap];print(c,gap,sum(bool(r['target_events']) for r in rs),len(rs))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
