"""Pre-pin and render complete Joe development presentations; no detector calls."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from synthetic_render_quality_v1 import audit_render, RIR_DELAYS, RIR_AMPLITUDES

ROOT=Path('research/wakeword_p4')
SOURCE=Path('.local/wakeword/joe-development-native-transport-v1')
PLAN=ROOT/'results/4090-joe-development-render-plan-v1.json'
OUT=Path('.local/wakeword/joe-development-render-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','scipy','soundfile']}


def make_plan():
    summary=json.loads((SOURCE/'summary.json').read_text())
    assert summary['mechanical_pack_qualified'] and summary['passed']==120
    records=SOURCE/'records.jsonl'
    assert sha(records)==summary['records_sha256']
    rows=[json.loads(s) for s in records.read_text().splitlines()]
    requests=[]
    for r in rows:
        assert r['audit']['passed'] and sha(SOURCE/r['path'])==r['sha256']
        for variant in range(3):
            seed=int(hashlib.sha256((r['id']+':joe-render-v1:'+str(variant)).encode()).hexdigest()[:16],16)
            requests.append(dict(id=r['id']+f'-v{variant}',source_id=r['id'],
                source_path=r['path'],source_sha256=r['sha256'],source_samples=r['samples'],
                target=r['target'],text=r['text'],length_scale=r['length_scale'],draw=r['draw'],
                variant=variant,seed=seed))
    plan=dict(schema='joe-complete-presentation-plan-v1',source_records_sha256=sha(records),
              source_summary_sha256=sha(SOURCE/'summary.json'),versions=versions(),
              renderer_sha256=sha(__file__),validator_sha256=sha(ROOT/'experiments/synthetic_render_quality_v1.py'),
              protocol_sha256=sha(ROOT/'experiments/joe_render_protocol_v1.md'),
              requests=requests,lead_samples=28800,tail_samples=16000,
              rir=dict(delays=RIR_DELAYS,amplitudes=RIR_AMPLITUDES),
              authorization='Joe development rendering only, no wake scoring or Mike inference')
    with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(requests=len(requests),plan_sha256=sha(PLAN))))


def render():
    plan=json.loads(PLAN.read_text())
    assert versions()==plan['versions'] and sha(__file__)==plan['renderer_sha256']
    assert sha(ROOT/'experiments/synthetic_render_quality_v1.py')==plan['validator_sha256']
    assert sha(ROOT/'experiments/joe_render_protocol_v1.md')==plan['protocol_sha256']
    assert sha(SOURCE/'records.jsonl')==plan['source_records_sha256']
    assert sha(SOURCE/'summary.json')==plan['source_summary_sha256']
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_bytes(PLAN.read_bytes())
    ir=np.zeros(961);ir[RIR_DELAYS]=RIR_AMPLITUDES
    rows=[]
    with (OUT/'records.jsonl').open('x') as log:
        for request in plan['requests']:
            row=dict(request)
            try:
                assert sha(SOURCE/row['source_path'])==row['source_sha256']
                source,rate=sf.read(SOURCE/row['source_path'],dtype='int16')
                assert rate==16000 and source.ndim==1 and len(source)==row['source_samples']
                dry=source.astype(np.float64)/32768
                variant=row['variant'];speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
                mixed=np.pad(speech,(plan['lead_samples'],plan['tail_samples']))
                snr=None if variant==0 else 25 if variant==1 else 15
                if variant:
                    noise=np.random.default_rng(row['seed']).normal(0,1,len(mixed))
                    noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2))
                    mixed+=noise
                gain=min(1.,.95/max(float(np.max(np.abs(mixed))),1e-8)) if variant else 1.
                mixed*=gain
                path=OUT/(row['id']+'.wav');sf.write(path,mixed,16000,subtype='PCM_16')
                pcm,out_rate=sf.read(path,dtype='int16');assert out_rate==16000
                row.update(path=path.name,sha256=sha(path),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                    samples=len(pcm),lead_samples=28800,tail_samples=16000,
                    native_generation_end=28800+len(source),keyword_end=28800+len(speech),
                    generated_output_end_sample=28800+len(source),full_rir_end_sample=28800+len(speech),
                    boundary_reference='Whole generated output, not human/wake-prefix phonetic endpoint',
                    rir_tail_samples=960 if variant==2 else 0,synthetic_rir=variant==2,
                    snr_db=snr,gain_after_mix=gain)
                row['audit']=audit_render(source,pcm,row)
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    summary=dict(plan_sha256=sha(PLAN),requested=len(plan['requests']),recorded=len(rows),
                 passed=sum(r.get('audit',{}).get('passed',False) for r in rows),
                 mechanically_qualified=len(rows)==360 and all(r.get('audit',{}).get('passed',False) for r in rows),
                 records_sha256=sha(OUT/'records.jsonl'),wake_model_scoring=False,mike_inference=False,
                 limits='Mechanical presentation integrity only; not phonetic truth or detector acceptance.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    if sys.argv[1:]==['--write-plan']:make_plan()
    elif sys.argv[1:]==['--render']:render()
    else:raise SystemExit('Use --write-plan or --render')
