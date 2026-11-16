"""Preregistered continuous mixed-session deployment-byte diagnostic; no tuning."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_multispeaker_formal_v1 import prepare, predict
from uint8_event_policy_v1 import event_steps

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-continuous-chatterbox-plan-v1.json'
OUT = Path('.local/wakeword/continuous-chatterbox-v1')
DATA = Path('.local/wakeword/chatterbox-development-v1')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def lock():
    models = [dict(name='c161', path='.local/wakeword/causal-source-balanced-int8-v1/int8.tflite', sha256='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'), dict(name='bc01', path='.local/wakeword/c161-chatterbox-int8-v1/int8.tflite', sha256='bc011e24ff04f70c3911d39b6cc027a016d68c6fd793124d777ebf5f29edefa8')]
    paths = [Path(__file__), Path(__file__).with_name('collect_multispeaker_formal_v1.py'), Path(__file__).with_name('uint8_event_policy_v1.py'), Path('research/wakeword_p4/training/features.py')]
    rows = []
    for i in range(36):
        meta = DATA / f'{i:03d}.json'
        row = json.loads(meta.read_text())
        assert row['index'] == i and row['request']['role'] == 'development-only'
        paths.append(meta)
        for item in row['files']:
            assert sha(item['path']) == item['sha256']
            paths.append(Path(item['path']))
        rows.append(row)
    for model in models:
        assert sha(model['path']) == model['sha256']
        paths.append(Path(model['path']))
    p = dict(models=models, rows=rows, order=[i for j in range(12) for i in (j,24+j,12+j)], gaps_samples=[8000,32000], prime_samples=81600, tail_samples=32000, threshold=.91796875, smoothing_steps=5, refractory_calls=67, rearm='half-threshold', files={str(p):sha(p) for p in paths}, versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']}, policy='Two gap conditions, all36 consumed builtin development clips once each per condition/model. One continuous zero prime; no frontend/model/event resets at clip boundaries. Trailing gap after every clip plus final tail. Event ownership is block-return container only, not phonetic truth or causal attribution. No fresh exposure claim, no FAR/acceptance/training/threshold selection. Both conditions reported without selection. CPU single-thread reference INT8 replay, not CPU training.')
    with PLAN.open('x') as f:
        f.write(json.dumps(p,indent=2)+'\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)]) == PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for path,h in p['files'].items():
        assert sha(path)==h,path
    for name,v in p['versions'].items():
        assert importlib.metadata.version(name)==v
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kw):
        return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    OUT.mkdir(exist_ok=False)
    results=[]
    for gap in p['gaps_samples']:
        chunks=[np.zeros(p['prime_samples'],np.int16)]
        timeline=[dict(kind='prime',start=-p['prime_samples'],end=0)]
        cursor=0
        for index in p['order']:
            row=p['rows'][index]
            wav,=[item['path'] for item in row['files'] if item['path'].endswith('.wav')]
            pcm,rate=sf.read(wav,dtype='int16')
            assert rate==16000 and pcm.ndim==1
            chunks.extend([pcm,np.zeros(gap,np.int16)])
            timeline.append(dict(kind='clip',index=index,group=row['request']['group'],start=cursor,end=cursor+len(pcm)))
            cursor+=len(pcm)
            timeline.append(dict(kind='gap',after=index,start=cursor,end=cursor+gap))
            cursor+=gap
        chunks.append(np.zeros(p['tail_samples'],np.int16))
        timeline.append(dict(kind='tail',start=cursor,end=cursor+p['tail_samples']))
        pcm=np.concatenate(chunks)
        values,support,_=prepare(features(pcm),len(pcm))
        clocks=np.minimum(((support+479)//480)*480,len(pcm))-p['prime_samples']
        record=dict(gap_samples=gap,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),samples=len(pcm),timeline=timeline,models={})
        for model in p['models']:
            _,codes,q=predict(values,model['path'],runtime)
            events=[]
            for step in event_steps(codes,p['threshold']):
                sample=int(clocks[step-1])
                owner=next(t for t in timeline if t['start']<sample<=t['end'])
                events.append(dict(step=step,relative_sample=sample,owner=owner))
            trace=OUT/f'{gap}-{model["name"]}.npz'
            np.savez_compressed(trace,uint8=codes,relative_block_return=clocks,support=support)
            clips=[e['owner']['index'] for e in events if e['owner']['kind']=='clip']
            groups={group:sorted(set(i for i in clips if p['rows'][i]['request']['group']==group)) for group in sorted(set(r['request']['group'] for r in p['rows']))}
            record['models'][model['name']]=dict(events=events,clip_containers_with_events=groups,trace_path=str(trace),trace_sha256=sha(trace),quantization=q)
        results.append(record)
        print(json.dumps(record),flush=True)
    (OUT/'result.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN),completed=True,results=results,threshold_selected=False,acceptance_accessed=False),indent=2)+'\n')

if __name__=='__main__':
    {'lock':lock,'run':run}[sys.argv[1]]()
