"""Whole native PCM matched to original anchor environments; gated planning."""
import argparse,hashlib,importlib.metadata,json,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
PLAN=R/'4090-hey-common-carrier-render-execution-v1.json';OUT=BASE/'hey-common-carrier-rendered-v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ('numpy','scipy','soundfile')}
def plan(check_only=False):
    needed=[R/'4090-hey-common-carrier-kokoro-generation-result-v1.json',R/'4090-hey-common-carrier-kokoro-independent-audit-v1.json']
    for path in needed:
        if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
    report,audit=[json.loads(p.read_text()) for p in needed]
    assert report['whole_pack_qualified'] and report['quality_pass']==report['cases']==1620
    assert audit['independently_qualified'] and audit['original_result_sha256']==sha(needed[0])
    if check_only:print(json.dumps(dict(status='ready',sources=1620)));return 0
    oldplan=R/'4090-hey-broad-render-plan-v1.json';old=json.loads(oldplan.read_text());schedule=Path(old['schedule']['path']);assert sha(schedule)==old['schedule']['sha256']
    sources={r['id']:r for r in map(json.loads,schedule.read_text().splitlines()) if r['id'].startswith('kokoro-negative-')};assert len(sources)==540
    records=BASE/'hey-common-carrier-kokoro-native-v1/records.jsonl';assert sha(records)==report['records_sha256']
    rows=[json.loads(l) for l in records.read_text().splitlines()];requests=[]
    for row in rows:
        req=row['request'];anchor=sources[f'kokoro-negative-{req["anchor_order"]:05}'];assert anchor['text']==req['anchor_text'] and anchor['engine']=='kokoro'
        assert anchor['pair_key']['conditioning']==req['conditioning_id'] and anchor['pair_key']['scale']==req['speed']
        f=row['files']['wav'];assert sha(f['path'])==f['sha256']
        requests.append(dict(order=req['order'],native=row,anchor=anchor,render_seeds=anchor['render_seeds']))
    p=dict(files={str(f):sha(f) for f in [Path(__file__),*needed,oldplan,schedule,records]},versions=versions(),requests=requests,
           lead_samples=48000,tail_samples=16000,rir_delays=old['rir_delays'],rir_amplitudes=old['rir_amplitudes'],
           policy='All1620native sources x original3conditions =4860whole rendered streams. Reuse anchor exact render seeds/quiet/noiseSNR25and15/lead48000/tail16000/fullRIR960/gain rule. Source length changes noise normalization/whole waveform duration; shared seed does not mean equal realized noise magnitude. No native speed change/trim/retry/filter. Any failure disqualifies all4860; no training or heldout access.',
           original_renderer_sha256=sha(Path(__file__).with_name('render_hey_broad_training_v1.py')))
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    assert versions()==p['versions'];OUT.mkdir(exist_ok=False)
    impulse=np.zeros(961);impulse[p['rir_delays']]=p['rir_amplitudes'];failures=[];count=0
    with (OUT/'records.jsonl').open('x') as log:
        for source in p['requests']:
            native=source['native'];path=Path(native['files']['wav']['path']);assert sha(path)==native['files']['wav']['sha256']
            pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==native['samples']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==native['pcm_sha256']
            dry=pcm.astype(np.float64)/32768
            for variant in range(3):
                speech=fftconvolve(dry,impulse,mode='full') if variant==2 else dry
                values=np.pad(speech,(48000,16000));snr=None if not variant else 25 if variant==1 else 15
                if variant:
                    noise=np.random.default_rng(source['render_seeds'][variant]).normal(size=len(values));noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2));values+=noise
                gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.;values*=gain
                wav=OUT/f'{count:04}.wav';sf.write(wav,values,16000,subtype='PCM_16');actual,rate=sf.read(wav,dtype='int16')
                checks=dict(finite=bool(np.isfinite(values).all()),full_length=len(actual)==len(pcm)+64000+(960 if variant==2 else 0),
                            format=rate==16000 and actual.ndim==1 and sf.info(wav).subtype=='PCM_16',no_fullscale=bool(np.all(np.abs(values)<1) and not np.any((actual==-32768)|(actual==32767))),
                            serialization_error=float(np.max(np.abs(actual.astype(np.float64)/32768-values)))<=1/32768+1e-7,
                            quiet_source_exact=variant!=0 or np.array_equal(actual[48000:48000+len(pcm)],pcm))
                if not all(checks.values()):failures.append(dict(order=count,checks=checks))
                row=dict(order=count,native_order=source['order'],request=native['request'],anchor_id=source['anchor']['id'],variant=variant,seed=source['render_seeds'][variant],
                         path=str(wav),bytes=wav.stat().st_size,sha256=sha(wav),pcm_sha256=hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest(),samples=len(actual),
                         lead_samples=48000,tail_samples=16000,rir_tail_samples=960 if variant==2 else 0,generated_container_end_sample=48000+len(speech),
                         generated_prefix_end_sample=None,snr_db=snr,gain_after_mix=gain,checks=checks,quality_pass=all(checks.values()))
                log.write(json.dumps(row)+'\n');log.flush();count+=1
            if (source['order']+1)%100==0:print(json.dumps(dict(sources=source['order']+1,rendered=count,total=4860)),flush=True)
    result=dict(execution_sha256=sha(PLAN),cases=count,quality_pass=count-len(failures),failures=failures,whole_pack_qualified=count==4860 and not failures,records_sha256=sha(OUT/'records.jsonl'),training_launched=False,limits=p['policy'])
    with (R/'4090-hey-common-carrier-render-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
