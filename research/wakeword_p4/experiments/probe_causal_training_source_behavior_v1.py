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
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-training-source-behavior-plan-v1.json';OUT=Path('.local/wakeword/causal-training-source-behavior-v1')
TEXTS={'Hey Nicks.','Hey Nicks!','Hey Nicks, turn on the light.','Hey Nicks, what time is it?','Hey Nick.','Hey Next.'}
def plan():
    parent=R/'4090-causal-sequence-formal-primed-plan-v1.json';p=json.loads(parent.read_text());native=Path('.local/wakeword/multispeaker-coverage-native-v1/records.jsonl');material=Path('.local/wakeword/causal-onset-material-v1/records.jsonl')
    rows=[json.loads(s) for s in native.read_text().splitlines()];rows=[r for r in rows if 'broad' in r['arms']]
    engines=sorted({r['voice'] for r in rows});speakers={e:min(r['speaker_id'] for r in rows if r['voice']==e) for e in engines}
    selected=[r for r in rows if r['speaker_id']==speakers[r['voice']] and r['text'] in TEXTS];ids={r['id'] for r in selected};assert len(selected)==90 and len(ids)==90
    requests=[]
    for line in material.read_text().splitlines():
        r=json.loads(line)
        if r['source']['id'] in ids and r['lead_samples']==28800:
            assert r['source_role']=='training' and r['source']['role']=='training';requests.append({k:r[k] for k in ['order','path','sha256','pcm_sha256','samples','container_end_samples','lead_samples','variant','source']})
    assert len(requests)==270 and {r['variant'] for r in requests}=={0,1,2}
    files=dict(p['files']);paths=[Path(__file__),parent,native,material,Path(__file__).with_name('collect_multispeaker_priming_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py')]
    for r in requests:
        assert sha(r['path'])==r['sha256'];paths.append(Path(r['path']))
    files.update({str(x):sha(x) for x in paths});model=next(m for m in p['models'] if m['condition']=='causal')
    lock=dict(schema='causal-training-source-behavior-plan-v1',files=files,model=model,requests=requests,speakers=speakers,native_rows=90,waveforms=270,threshold=.91796875,prime_samples=PRIME,expected_new_gib=.05,missing_current_material=['Muse: no native utterance in current training pack','Nick/Next command-carrier utterances absent; bare complete utterances only'],policy='Mac6885456303 behavioral question, not numerical/tensor/gradient audit. All existing requested Nyx/Nick/Next whole utterances for minimum numeric training speakerID per each ARU/Arctic engine, all original speeds/draws, all3quiet/noise/fullRIR at existing1.8lead. Reuse exact saved trainingmaterial PCM only, no TTS/regeneration/newvoice/newdev/acceptance. Actual existing81600zeroPCM→wholePCM continuousfrontend/model/event path, fixed reference.91796875/smoothing5/67cooldown; no threshold search. Record fullrawcodes and allprime/app events, generated wholecontainer clocks; no invented wake-prefix endpoint. This stage cannot answer absent Muse or Nick/Next commands; record gap rather than substitute other sources or claim independent/generalization success. No training or candidate promotion.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),native_rows=90,waveforms=270,speakers=speakers,missing=lock['missing_current_material'])))
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
        result=dict(schema='causal-training-source-behavior-original-v1',plan_sha256=sha(LOCK),launch=launch,records=records,groups=groups,waveforms=len(records),missing_current_material=p['missing_current_material'],qualification_claim=False,threshold_selected=False,policy=p['policy'])
        with (R/'4090-causal-training-source-behavior-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,waveforms=len(records),groups=groups)),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

def common_pcm_sha(pcm):
    import hashlib
    return hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
