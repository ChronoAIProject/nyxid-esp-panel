"""Fixed-profile admitted training84 census; all three models, no selection."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-training-carrier-replay-plan-v1.json'
OUT = Path('.local/wakeword/training-carrier-replay-v1')
MODELS = dict(c161='.local/wakeword/causal-source-balanced-int8-v1/int8.tflite',
              silence='.local/wakeword/context-silence-int8-v1/int8.tflite',
              speech='.local/wakeword/context-speech-int8-v1/int8.tflite')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():
    admission = R / '4090-chatterbox-training-admission-v1.json'
    rows = json.loads(admission.read_text())['admitted']
    assert len(rows) == 84
    paths = [Path(__file__), admission] + list(map(Path, MODELS.values()))
    paths += [Path(__file__).with_name(n) for n in ['collect_multispeaker_formal_v1.py','uint8_event_policy_v1.py']]
    paths.append(Path('research/wakeword_p4/training/features.py'))
    for row in rows:
        p = Path(row['wav']['path'])
        assert p.stat().st_size == row['wav']['bytes'] and sha(p) == row['wav']['sha256']
        paths.append(p)
    record = dict(files={str(p):sha(p) for p in paths}, models=MODELS, rows=rows,
        versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},
        profile=dict(threshold=.91796875,smoothing=5,refractory=67,prime_samples=81600,native_return_samples=480),
        policy='All84 admitted training-only clips, no quarantined examples. All3 models in fixed order; no threshold/checkpoint/calibration selection. Training fit diagnostic only, no generalization, phoneme truth or acceptance. Continuous prime, no reset at audio boundary; original full PCM without tail extension. Prefix-end timestamps automatic; report signed timing descriptively. CPU1thread deployment reference replay, not CPU training.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)]) == PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    import shutil
    assert shutil.disk_usage('.').free > 100*1024**3
    import numpy as np
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    from collect_multispeaker_formal_v1 import prepare,predict
    from uint8_event_policy_v1 import event_steps
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    def runtime(**kwargs):
        return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kwargs)
    OUT.mkdir(exist_ok=False)
    records=[]
    for row in p['rows']:
        pcm,rate=sf.read(row['wav']['path'],dtype='int16')
        assert rate==16000 and pcm.ndim==1
        combined=np.concatenate((np.zeros(81600,np.int16),pcm))
        values,support,_=prepare(features(combined),len(combined))
        blocks=np.minimum(((support+479)//480)*480,len(combined))-81600
        for condition,model in p['models'].items():
            _,codes,q=predict(values,model,runtime)
            events=[dict(step=s,relative_sample=int(blocks[s-1]),finish=bool(support[s-1]>len(combined))) for s in event_steps(codes,.91796875)]
            trace=OUT/f'{row["index"]:03d}-{condition}.npz'
            np.savez_compressed(trace,uint8=codes,relative_block_return=blocks,support=support)
            prefix=row.get('automatic_prefix_end_seconds')
            result=dict(index=row['index'],condition=condition,request=row['request'],
                automatic_prefix_end_seconds=prefix,events=events,
                real_events=[e for e in events if e['relative_sample']>0],
                prime_events=[e for e in events if e['relative_sample']<=0],
                trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),quantization=q)
            if prefix is not None:
                result['signed_first_event_from_automatic_prefix_ms']=next((e['relative_sample']/16-prefix*1000 for e in result['real_events']),None)
            records.append(result)
            (OUT/f'{row["index"]:03d}-{condition}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(index=row['index'],condition=condition,events=len(result['real_events']))),flush=True)
    (OUT/'result.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN),records=records,training_diagnostic=True),indent=2)+'\n')
if __name__=='__main__':
    {'plan':plan,'run':run}[sys.argv[1]]()
