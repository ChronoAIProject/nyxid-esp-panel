"""Existing zero-PCM priming, one continuous frontend/model per consumed input."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from collect_multispeaker_formal_v1 import prepare,predict
from collect_complete_development_v1 import runtime_identity_files
R=Path('research/wakeword_p4/results')
LOCK=R/'4090-multispeaker-priming-plan-v1.json'
OUT=Path('.local/wakeword/multispeaker-priming-v1')
PRIME=((309+200)*160+479)//480*480
assert PRIME==81600

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def plan():
    parent=R/'4090-multispeaker-formal-development-execution-v1.json'
    p=json.loads(parent.read_text());files=dict(p['files'])
    paths=[parent,Path(__file__),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('collect_complete_development_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py'),Path('research/wakeword_p4/training/features.py')]
    paths+=list(Path(__file__).with_name('runtime_priming_sources_v1').iterdir())+runtime_identity_files()
    files.update({str(path):sha(path) for path in paths})
    for path,h in files.items():assert sha(path)==h,path
    lock=dict(schema='multispeaker-existing-priming-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},models=p['models'],materials=p['materials'],grid=p['grid'],prime_samples_requested=81440,prime_samples_actual=PRIME,source_commit='70ad0ea307d366163b0253d06335b623c288e411',expected_new_gib=.5,retries=0,policy='Consumed-only existing zero-PCM priming: one fresh continuous frontend and model for 81600 zero samples plus unchanged whole original PCM. No transition reset, feature splice, guard, readiness delay, score selection or training. Keep every priming UINT8 output; event policy spans transition, prime events ignored only by app. Original complete-input final160hop/feature-realzero finish retained ONLY at end of whole original stream, not priming boundary. Preserve full real exposure, first phrases,138points and old cold/readiness failures. Record support and480block-return clocks separately; final finish calls explicitly marked. Research diagnosis only, v30 retained, no freshdev/acceptance.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),prime_samples=PRIME)))

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert importlib.metadata.version(name)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    OUT.mkdir(exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import soundfile as sf
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from features import features
        def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
        records={}
        for m in p['models']:
            name=m['target']+'-'+m['condition'];records[name]=[]
            for phase in p['materials']:(OUT/name/phase).mkdir(parents=True)
        for phase,material in p['materials'].items():
            for i,row in enumerate(json.loads(Path(material['path']).read_text())['records']):
                wav=row['path'] if phase=='synthetic' else row['wav_path'];pcm,rate=sf.read(wav,dtype='int16')
                assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                # Fresh extraction across concatenated PCM is a single frontend
                # lifetime; features() pads only the final real-stream hop.
                combined=np.concatenate((np.zeros(PRIME,np.int16),pcm));raw=features(combined)
                values,support,clock=prepare(raw,len(combined))
                prime_calls=int(np.sum(support<=PRIME));assert prime_calls==169
                # 508 frontend frames at prime end => 169 calls + one
                # pending frame, which is retained in the continuous array.
                assert (PRIME//160-2)%3==1
                blocks=((support+479)//480)*480
                finish=blocks>len(combined)
                blocks=np.minimum(blocks,len(combined))
                for m in p['models']:
                    name=m['target']+'-'+m['condition']
                    scores,codes,q=predict(values,m['path'],runtime)
                    trace=OUT/name/phase/f'{i:06}.npz'
                    np.savez_compressed(trace,uint8=codes,probability=scores,support_unclamped=support,sample_end=clock,block_return_sample=blocks,finish_call=finish,real_support_sample=support-PRIME,real_block_return_sample=blocks-PRIME)
                    rec=dict(model=name,phase=phase,index=i,record=row,trace_path=str(trace),trace_sha256=sha(trace),quantization=q,raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),steps=len(codes),prime_calls=prime_calls,pending_feature_frames_at_transition=1,prime_samples=PRIME,original_real_samples=len(pcm),continuous_frontend=True,continuous_model=True)
                    records[name].append(rec)
                    with (OUT/name/'records.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                    print(json.dumps({k:rec[k] for k in ['model','phase','index','steps','prime_calls']}),flush=True)
        reports=[]
        for m in p['models']:
            name=m['target']+'-'+m['condition'];assert len(records[name])==372
            path=OUT/name/'report.json';path.write_text(json.dumps(dict(model=m,execution_sha256=sha(LOCK),records=records[name],all_passed=True,threshold_selected=False),indent=2)+'\n');reports.append(dict(model=name,path=str(path),sha256=sha(path)))
        result=dict(schema='multispeaker-existing-priming-original-v1',execution_sha256=sha(LOCK),launch=launch,models=reports,all_passed=True,threshold_selected=False,acceptance_scored=False,policy=p['policy'])
        (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,models=2,traces=744)),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
