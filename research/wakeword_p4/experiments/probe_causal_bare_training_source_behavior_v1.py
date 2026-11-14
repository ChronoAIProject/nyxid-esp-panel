"""Small actual primed recognition probe on immutable current training waveforms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from collect_multispeaker_priming_v1 import sha,PRIME,prepare,predict
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-bare-training-source-behavior-plan-v1.json';OUT=Path('.local/wakeword/causal-bare-training-source-behavior-v1')
TEXTS={'Hey Nicks.','Hey Nicks!','Hey Nicks, turn on the light.','Hey Nicks, what time is it?','Hey Nick.','Hey Next.'}
def plan():
    parent=R/'4090-causal-training-source-behavior-plan-v1.json';p=json.loads(parent.read_text())
    original=R/'4090-causal-bare-positive-int8-original-v1.json';r=json.loads(original.read_text());m=r['model'];assert sha(m['path'])==m['sha256']
    p['schema']='causal-bare-training-source-behavior-plan-v1'
    p['model']=dict(target='hey_nyx',condition='bare-positive',path=m['path'],sha256=m['sha256'])
    paths=[Path(__file__),parent,original,Path(m['path'])]
    p['files'].update({str(x):sha(x) for x in paths})
    p['policy']='Fixed new bare-positive model on the identical270existing training-source waveforms selected before the prior probe. No new TTS/cropping/training/thresholdsearch/freshroles. Originalcommands were excluded from new positive training, but sources/voices are not independent. Actual81600continuousprime/reference.91796875. Compare wholeutterance any-after-source hit/miss and bareconfusion with retained prior probe; no invented within-command wake boundary. Determine whether light recognition deficits also occur on existing source voices before proposing another training intervention. No qualification claim.'
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),waveforms=p['waveforms'],model=p['model'])))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import soundfile as sf
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
        def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
        records=[]
        for i,row in enumerate(p['requests']):
            pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
            assert common_pcm_sha(pcm)==row['pcm_sha256']
            combined=np.concatenate((np.zeros(PRIME,np.int16),pcm));raw=features(combined);values,support,clock=prepare(raw,len(combined));scores,codes,q=predict(values,p['model']['path'],runtime)
            all_steps=event_steps(codes,p['threshold']);total=len(combined);pairs=[dict(step=s,sample=min(((3*s+2)*160+479)//480*480,total)-PRIME,support_relative_sample=(3*s+2)*160-PRIME) for s in all_steps];prime=[e for e in pairs if e['sample']<=0];app=[e for e in pairs if e['sample']>0]
            for e in app:e['generated_whole_container_signed_ms']=(e['sample']-row['container_end_samples'])/16
            trace=OUT/f'{i:04}.npz';np.savez_compressed(trace,uint8=codes,support_unclamped=support,sample_end=clock)
            observed=dict(index=i,record=row,trace_path=str(trace),trace_sha256=sha(trace),quantization=q,prime_events=prime,app_events=app,any_after_source=any(e['sample']>=row['lead_samples'] for e in app),pre_source_events=sum(e['sample']<row['lead_samples'] for e in app),false_clip=bool(app) if row['source']['label']==0 else None)
            records.append(observed);print(json.dumps(dict(index=i,text=row['source']['text'],engine=row['source']['engine'],variant=row['variant'],app_events=len(app),any_after_source=observed['any_after_source'])),flush=True)
        groups=[]
        for engine in sorted(p['speakers']):
            for text in sorted(TEXTS):
                a=[r for r in records if r['record']['source']['engine']==engine and r['record']['source']['text']==text];assert a
                groups.append(dict(engine=engine,text=text,clips=len(a),label=a[0]['record']['source']['label'],after_source_hits=sum(r['any_after_source'] for r in a),false_clips=sum(bool(r['false_clip']) for r in a),pre_source_events=sum(r['pre_source_events'] for r in a)))
        result=dict(schema='causal-bare-training-source-behavior-original-v1',plan_sha256=sha(LOCK),launch=launch,records=records,groups=groups,waveforms=len(records),missing_current_material=p['missing_current_material'],qualification_claim=False,threshold_selected=False,policy=p['policy'])
        with (R/'4090-causal-bare-training-source-behavior-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,waveforms=len(records),groups=groups)),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

def common_pcm_sha(pcm):
    import hashlib
    return hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
