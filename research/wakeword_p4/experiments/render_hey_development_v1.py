"""Lock and render full reserved development PCM; no wake-model inference."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
from generate_hey_development_native_v1 import OUT as NATIVE,CONTRACT,sha
from generate_hey_paired_material_v2 import versions
from synthetic_output_quality_v1 import audit_output
from synthetic_development_render_quality_v1 import audit_render,RIR_DELAYS,RIR_AMPLITUDES

PLAN=Path('research/wakeword_p4/results/4090-hey-development-render-plan-v1.json')
DEST=Path('.local/wakeword/hey-paired-development-rendered-v1')


def register():
    summary=json.loads((NATIVE/'summary.json').read_text())
    assert summary['pack_qualified'] and summary['recorded']==summary['passed']==120
    assert summary['records_sha256']==sha(NATIVE/'records.jsonl')
    contract=json.loads(CONTRACT.read_text());render=contract['rendering']
    assert render['rir_delays']==RIR_DELAYS and render['rir_amplitudes']==RIR_AMPLITUDES
    paths=[Path(__file__),CONTRACT,NATIVE/'summary.json',NATIVE/'records.jsonl']
    paths += [Path(__file__).with_name(n) for n in ('generate_hey_development_native_v1.py','generate_hey_paired_material_v2.py','synthetic_output_quality_v1.py','synthetic_development_render_quality_v1.py')]
    plan=dict(schema='hey-development-render-plan-v1',files={str(p):sha(p) for p in paths},
              versions=versions(),rendering=render,requests=360,
              authorization='Development rendering only, full source and full RIR tail; no detector inference or acceptance access.',
              failure_policy='Whole pack rejected on any audit failure; no filtering/retries/replacements.')
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),rendered=False)))


def render():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes()
    plan=json.loads(remote);assert versions()==plan['versions']
    for p,h in plan['files'].items():assert sha(p)==h
    DEST.mkdir(exist_ok=False);(DEST/'plan.json').write_bytes(remote)
    ir=np.zeros(1233);ir[RIR_DELAYS]=RIR_AMPLITUDES
    count=passed=0
    with (DEST/'records.jsonl').open('x') as log:
        for line in (NATIVE/'records.jsonl').open():
            source=json.loads(line)
            assert source['role']=='development' and audit_output(NATIVE,source)['passed']
            pcm,rate=sf.read(NATIVE/source['path'],dtype='int16');assert rate==16000
            dry=pcm.astype(np.float64)/32768
            for variant in range(3):
                row=dict(source,variant=variant,seed=source['render_seeds'][variant],
                         source_samples=len(pcm),source_path=str(NATIVE/source['path']),
                         source_sha256=source['sha256'],source_pcm_sha256=__import__('hashlib').sha256(pcm.astype('<i2').tobytes()).hexdigest())
                speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
                values=np.pad(speech,(28800,16000));snr=None if variant==0 else 25 if variant==1 else 15
                if variant:
                    noise=np.random.default_rng(row['seed']).normal(0,1,len(values))
                    noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2))
                    values+=noise
                gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.
                values*=gain
                path=DEST/(source['id']+f'-n{variant}.wav');sf.write(path,values,16000,subtype='PCM_16')
                actual,rate=sf.read(path,dtype='int16');assert rate==16000
                row.update(path=path.name,sha256=sha(path),samples=len(actual),native_generation_end=28800+len(pcm),keyword_end=28800+len(speech),boundary_kind='Full generated container including command, not acoustic wake endpoint',rir_tail_samples=1232 if variant==2 else 0,lead_samples=28800,tail_samples=16000,snr_db=snr,synthetic_rir=variant==2,gain_after_mix=gain)
                row['render_audit']=audit_render(pcm,actual,row)
                count+=1;passed+=int(row['render_audit']['passed'])
                log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(dict(recorded=count,passed=passed)),flush=True)
    summary=dict(schema='hey-development-render-completion-v1',plan_sha256=sha(PLAN),requested=360,recorded=count,passed=passed,pack_qualified=count==passed==360,records_sha256=sha(DEST/'records.jsonl'),wake_model_scoring=False,acceptance_generated=False,limits='Rendered mechanical quality only; token timing/source history and scoring admission pending.')
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','render'));a=p.parse_args()
    {'register':register,'render':render}[a.action]()
