"""Phase-matched silence-history counterfactual, development diagnostic only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_multispeaker_formal_v1 import prepare,predict
from uint8_event_policy_v1 import event_steps
from replay_continuous_chatterbox_v1 import PLAN as SOURCE_PLAN,sha

R=Path('research/wakeword_p4/results')
PLAN=R/'4090-continuous-context-plan-v1.json'
SOURCE=R/'4090-continuous-chatterbox-original-v1.json'
OUT=Path('.local/wakeword/continuous-context-v1')

def lock():
    old=json.loads(SOURCE_PLAN.read_text())
    files=dict(old['files'])
    for f in [SOURCE_PLAN,SOURCE,Path(__file__)]:files[str(f)]=sha(f)
    p=dict(files=files,source_plan=str(SOURCE_PLAN),source_result=str(SOURCE),versions=old['versions'],policy='All12 Nyx clips x2 gap-phase schedules x2 fixed models. Replace all preceding speech with zero PCM, preserve target start modulo480 using prefix81600+(session_start%480). Continuous frontend and model within each counterfactual. Native480 clocks, fixed original event policy, complete target and32000zero tail. This removes prior audio jointly from frontend and model; does not isolate which state causes effects. Compare target containers only plus report all events. No training, threshold change, fresh exposure, acceptance or deployment reset recommendation.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    import importlib.metadata
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    source=json.loads(SOURCE_PLAN.read_text());sessions=json.loads(SOURCE.read_text())['results']
    OUT.mkdir(exist_ok=False);records=[]
    for session in sessions:
        for t in session['timeline']:
            if t['kind']!='clip' or t['group']!='nyx':continue
            index=t['index'];row=source['rows'][index]
            wav,=[i['path'] for i in row['files'] if i['path'].endswith('.wav')]
            pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1
            prefix=81600+t['start']%480
            full=np.concatenate([np.zeros(prefix,np.int16),pcm,np.zeros(32000,np.int16)])
            values,support,_=prepare(features(full),len(full))
            clocks=np.minimum(((support+479)//480)*480,len(full))-prefix
            record=dict(gap_samples=session['gap_samples'],index=index,original_start=t['start'],phase=t['start']%480,prefix_samples=prefix,pcm_sha256=hashlib.sha256(full.astype('<i2').tobytes()).hexdigest(),models={})
            for model in source['models']:
                name=model['name'];_,codes,q=predict(values,model['path'],runtime)
                events=[dict(step=s,relative_sample=int(clocks[s-1])) for s in event_steps(codes,.91796875)]
                trace=OUT/f'{session["gap_samples"]}-{index:03d}-{name}.npz'
                np.savez_compressed(trace,uint8=codes,relative_block_return=clocks,support=support)
                mask=(clocks>0)&(clocks<=len(pcm))
                means=np.concatenate([np.full(4,np.nan),np.convolve(codes.astype(np.int64),np.ones(5,np.int64),'valid')/1280.0])
                previous=session['models'][name]
                record['models'][name]=dict(events=events,target_events=sum(0<e['relative_sample']<=len(pcm) for e in events),continuous_target_events=sum(e['owner']==t for e in previous['events']),target_max_mean=float(np.nanmax(means[mask])),trace_path=str(trace),trace_sha256=sha(trace),quantization=q)
            records.append(record)
            print(json.dumps(dict(gap=session['gap_samples'],index=index)),flush=True)
    (OUT/'result.json').write_text(json.dumps(dict(completed=True,plan_sha256=sha(PLAN),records=records,threshold_changed=False),indent=2)+'\n')

if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
