"""Independent sparse-RIR saved-PCM replay audit; no TTS or detector inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf

RESULTS=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword/hey-broad-training-rendered-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--starttime');a=p.parse_args()
    if a.wait_pid:
        assert a.starttime;proc=Path(f'/proc/{a.wait_pid}/stat')
        while proc.exists():
            try:s=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if s!=a.starttime:break
            time.sleep(5)
    pp=RESULTS/'4090-hey-broad-render-plan-v1.json';rp=RESULTS/'4090-hey-broad-render-v1.json'
    d=json.loads(rp.read_text());plan=json.loads(pp.read_text());assert sha(pp)==d['plan_sha256']
    records=BASE/'records.jsonl';assert sha(records)==d['records_sha256']
    sp=Path(plan['schedule']['path']);assert sha(sp)==plan['schedule']['sha256'];sources=[json.loads(s) for s in sp.read_text().splitlines()]
    previous=None;source_pcm=None;count=0;bad=[];geom_bad=[];max_error=0.
    with records.open() as stream:
        for line in stream:
            r=json.loads(line);source=sources[count//3];variant=count%3
            assert r['order']==count and r['source']==source and r['variant']==variant and r['seed']==source['render_seeds'][variant]
            if previous!=source['id']:
                sourcepath=Path(source['source_path']);assert sha(sourcepath)==source['source_sha256']
                source_pcm,rate=sf.read(sourcepath,dtype='int16');assert rate==16000 and len(source_pcm)==source['source_samples']
                assert hashlib.sha256(source_pcm.astype('<i2').tobytes()).hexdigest()==source['source_pcm_sha256'];previous=source['id']
            dry=source_pcm.astype(np.float64)/32768
            if variant==2:
                speech=np.zeros(len(dry)+960,dtype=np.float64)
                for delay,amp in zip(plan['rir_delays'],plan['rir_amplitudes']):speech[delay:delay+len(dry)]+=amp*dry
            else:speech=dry
            expected=np.pad(speech,(48000,16000))
            if variant:
                noise=np.random.default_rng(source['render_seeds'][variant]).standard_normal(len(expected))
                target_rms=np.linalg.norm(dry)/np.sqrt(len(dry))*10**(-(25 if variant==1 else 15)/20)
                noise*=target_rms/(np.linalg.norm(noise)/np.sqrt(len(noise)));expected+=noise
            gain=min(1.,.95/max(float(np.max(np.abs(expected))),1e-8)) if variant else 1.;expected*=gain
            wav=Path(r['path']);assert sha(wav)==r['sha256'];pcm,rate=sf.read(wav,dtype='int16');info=sf.info(wav)
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
            error=float(np.max(np.abs(pcm.astype(np.float64)/32768-expected))) if len(pcm)==len(expected) else float('inf');max_error=max(max_error,error)
            good=(len(pcm)==len(expected)==r['samples'] and rate==16000 and info.subtype=='PCM_16' and pcm.ndim==1 and
                np.isfinite(expected).all() and np.max(np.abs(expected))<1 and np.min(pcm)>-32768 and np.max(pcm)<32767 and
                error<=1/32768+1e-7 and abs(gain-r['gain_after_mix'])<=1e-12 and
                r['generated_container_end_sample']==48000+len(speech) and r['rir_tail_samples']==(960 if variant==2 else 0))
            if variant==0:good=good and np.array_equal(pcm[48000:48000+len(source_pcm)],source_pcm) and not np.any(pcm[:48000]) and not np.any(pcm[-16000:])
            assert bool(good)==r['quality_pass']
            if not good:bad.append(count)
            if source['kind']=='positive':
                end=48000+source['generated_prefix_end']+(960 if variant==2 else 0)
                assert r['generated_prefix_end_sample']==end
                lead_frame=48000//160-2;keyword_frame=end//160-2
                if keyword_frame+27-225>lead_frame:geom_bad.append(count)
            else:assert r['generated_prefix_end_sample'] is None
            count+=1
            if count%5000==0:print(json.dumps(dict(audited=count,total=106956)),flush=True)
    assert count==d['cases']==106956 and d['quality_pass']==count-len(bad)
    result=dict(schema='hey-broad-render-independent-audit-v1',script_sha256=sha(Path(__file__)),original_report_sha256=sha(rp),
        cases=count,sources=len(sources),independently_qualified=count-len(bad),failures=bad,
        context225_full_source_prefix_failures=geom_bad,maximum_pcm_error=max_error,
        audit='Sparse direct4tap RIR versus original FFT, independently reconstructed gain/noise/full source+tail and exact PCM/quiet source; all generated prefix clocks/context225 maxjitter27 checked.',
        limits='Synthetic process/internal generated timing only, no phonetic truth or trained-model performance; Gaussian RNG algorithm shared, source histories/pretraining not certified here.')
    with (RESULTS/'4090-hey-broad-render-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
