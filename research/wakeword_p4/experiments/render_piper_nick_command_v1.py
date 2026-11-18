"""Render every admitted broad training source once per locked condition."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-piper-nick-command-render-plan-v1.json';SCHEDULE=BASE/'piper-nick-command-render-schedule-v1.jsonl'
OUT=BASE/'piper-nick-command-rendered-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['numpy','scipy','soundfile']}
def sources():
    reportpath=RESULTS/'4090-piper-nick-command-material-original-v1.json'
    auditpath=RESULTS/'4090-piper-nick-command-material-review-v1.json'
    receipt=RESULTS/'4090-piper-nick-command-material-execution-v1.json'
    report=json.loads(reportpath.read_text());audit=json.loads(auditpath.read_text())
    assert audit['material_admitted'] and audit['report_sha256']==sha(reportpath)
    assert json.loads(receipt.read_text())['actual_exit_code']==0
    records=BASE/'piper-nick-command-material-v1/records.jsonl'
    assert sha(records)==report['records_sha256']
    rows=[]
    for r in map(json.loads,records.read_text().splitlines()):
        assert r['quality_pass'];request=r['request'];anchor=request['positive_anchor']
        wav=BASE/f"piper-nick-command-material-v1/{r['order']:05}.wav"
        f=r['files'][str(wav)];assert sha(wav)==f['sha256']
        rows.append(dict(order=r['order'],id=f"piper-nick-command-{r['order']:05}",kind='negative',target='common_negative',engine='piper',pair_key=anchor['pair_key'],text=request['text'],source_path=str(wav),source_sha256=f['sha256'],source_pcm_sha256=r['pcm_sha256'],source_samples=r['samples'],generated_prefix_end=None,render_seeds=anchor['render_seeds'],positive_anchor=anchor))
    assert len(rows)==6048
    return rows,{str(p):sha(p) for p in [reportpath,auditpath,receipt,records,Path(__file__).with_name('render_hey_broad_training_v1.py')]}

def plan():
    assert not SCHEDULE.exists();rows,files=sources()
    SCHEDULE.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    decision=RESULTS/'4090-hey-broad-context-decision-v1.json'
    d=dict(schema='piper-nick-command-render-plan-v1',source_sha256=sha(Path(__file__)),files=files,context_decision_sha256=sha(decision),versions=versions(),
        schedule=dict(path=str(SCHEDULE),sha256=sha(SCHEDULE),sources=6048,rendered=18144),
        lead_samples=48000,tail_samples=16000,conditions=['quiet','25dB_noise','15dB_noise_full_RIR'],
        rir_delays=[0,160,480,960],rir_amplitudes=[1.,.25,-.12,.06],
        policy='Full original serialized PCM16 source, no speed change/trim; full RIR tail, Gaussian noise from fixed seeds at dry-source RMS reference, original min(1,.95/peak) attenuation after noisy mixing only; preserve all rendered failures. Paired positives share noise seed per variant, common negatives shared across arms.',
        lead_change='48000 samples leaves at least context225 before source onset; explicit change versus pilot28800, not a word/voice-only isolated intervention.',
        boundary='Original source sample0 through saved generated wake-prefix end + full RIR tail; Qwen full bare clip. No token-start relaxation or invented phonetic boundary.',
        stage_policy='Training-only material rendering, no model scoring or training; heldout untouched.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('6048 training-only Nick sources,18144 presentations locked')

def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and versions()==d['versions']
    for p,h in d['files'].items():assert sha(Path(p))==h
    assert sha(RESULTS/'4090-hey-broad-context-decision-v1.json')==d['context_decision_sha256']
    schedule=Path(d['schedule']['path']);assert sha(schedule)==d['schedule']['sha256']
    assert not OUT.exists();OUT.mkdir();ir=np.zeros(961);ir[d['rir_delays']]=d['rir_amplitudes']
    count=0;failures=[]
    with (OUT/'records.jsonl').open('x') as log:
        for source in map(json.loads,schedule.read_text().splitlines()):
            path=Path(source['source_path']);assert sha(path)==source['source_sha256'];pcm,rate=sf.read(path,dtype='int16')
            assert rate==16000 and pcm.ndim==1 and len(pcm)==source['source_samples']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==source['source_pcm_sha256']
            dry=pcm.astype(np.float64)/32768
            for variant in range(3):
                speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
                values=np.pad(speech,(d['lead_samples'],d['tail_samples']));snr=None if variant==0 else 25 if variant==1 else 15
                if variant:
                    noise=np.random.default_rng(source['render_seeds'][variant]).normal(0,1,len(values))
                    noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2));values+=noise
                gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.;values*=gain
                wav=OUT/f'{count:06}.wav';sf.write(wav,values,16000,subtype='PCM_16');actual,sr=sf.read(wav,dtype='int16')
                checks=dict(finite=bool(np.isfinite(values).all()),full_length=len(actual)==len(pcm)+48000+16000+(960 if variant==2 else 0),
                    format=sr==16000 and actual.ndim==1 and sf.info(wav).subtype=='PCM_16',
                    no_fullscale=bool(np.max(np.abs(values))<1 and np.min(actual)>-32768 and np.max(actual)<32767),
                    serialization_error=float(np.max(np.abs(actual.astype(np.float64)/32768-values)))<=1/32768+1e-7,
                    quiet_source_exact=variant!=0 or np.array_equal(actual[48000:48000+len(pcm)],pcm))
                good=all(checks.values())
                if not good:failures.append(dict(order=count,checks=checks))
                prefix=48000+source['generated_prefix_end']+(960 if variant==2 else 0) if source['kind']=='positive' else None
                row=dict(order=count,source=source,variant=variant,seed=source['render_seeds'][variant],path=str(wav),sha256=sha(wav),
                    pcm_sha256=hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest(),samples=len(actual),lead_samples=48000,tail_samples=16000,
                    rir_tail_samples=960 if variant==2 else 0,snr_db=snr,gain_after_mix=gain,generated_prefix_end_sample=prefix,
                    generated_container_end_sample=48000+len(speech),checks=checks,quality_pass=good)
                log.write(json.dumps(row)+'\n');log.flush();count+=1
            if (source['order']+1)%200==0:print(json.dumps(dict(sources=source['order']+1,rendered=count,total=18144)),flush=True)
    result=dict(schema='piper-nick-command-render-result-v1',plan_sha256=sha(PLAN),records_sha256=sha(OUT/'records.jsonl'),cases=count,
        quality_pass=count-len(failures),failures=failures,training_launched=False,independent_full_render_audit_pending=True)
    with (RESULTS/'4090-piper-nick-command-render-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
