"""Independent saved speech-negative audit; no model imports or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True,choices=['piper','kokoro']);p.add_argument('--wait-pid',type=int);p.add_argument('--starttime');a=p.parse_args()
    if a.wait_pid:
        assert a.starttime;proc=Path(f'/proc/{a.wait_pid}/stat')
        while proc.exists():
            try:s=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if s!=a.starttime:break
            time.sleep(5)
    report=RESULTS/f'4090-hey-broad-common-{a.engine}-generation-v1.json';d=json.loads(report.read_text())
    pp=RESULTS/'4090-hey-broad-common-negative-generation-plan-v1.json';plan=json.loads(pp.read_text())
    assert sha(pp)==d['plan_sha256'];sp=Path(plan['schedule']['path']);assert sha(sp)==plan['schedule']['sha256']
    expected=[r for r in map(json.loads,sp.read_text().splitlines()) if r['engine']==a.engine]
    out=BASE/f'hey-broad-common-{a.engine}-native-v1';records=out/'records.jsonl';assert sha(records)==d['records_sha256']
    rows=[json.loads(s) for s in records.read_text().splitlines()];assert len(rows)==len(expected)==d['cases']
    count=0;reused=0;failures=[]
    oldrows=[json.loads(s) for s in (BASE/'piper-balanced-training-full-v1/manifest.jsonl').read_text().splitlines()]
    for r,request in zip(rows,expected):
        assert r['order']==request['order'] and r['request']==request
        key=a.engine+'|'+(request['lang']+'|' if a.engine=='kokoro' else '')+request['text']
        meta=plan['text_inventory'][key];assert r['metadata']==meta
        arrays={}
        for name,f in r['files'].items():
            path=Path(f['path']);assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'];count+=1
            if name!='wav':arrays[name]=np.load(path,allow_pickle=False)
        if request['action']=='reuse':
            old=oldrows[request['source_order']];assert old['token_ids']==meta['token_ids'] and old['phonemes']==meta['phonemes'];reused+=1
        native=arrays['native'];resampled=arrays['resampled'];dur=arrays['durations'].ravel()
        numerator,denominator=(320,441) if a.engine=='piper' else (2,3)
        assert native.ndim==resampled.ndim==1 and native.dtype==resampled.dtype==np.float32
        assert np.array_equal(resample_poly(native,numerator,denominator),resampled)
        assert len(dur)==len(meta['token_ids']) and np.isfinite(dur).all() and np.all(dur>=0) and np.array_equal(dur,dur.astype(np.int64))
        assert sum(map(int,dur))*(1 if a.engine=='piper' else 600)==len(native)
        if a.engine=='kokoro':
            assert arrays['tokens'].ravel().tolist()==meta['token_ids']
            assert arrays['speed'].dtype==np.float32 and arrays['speed'][0]==np.float32(request['scale'])
        wav=Path(r['files']['wav']['path']);pcm,rate=sf.read(wav,dtype='int16');info=sf.info(wav)
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
        good=(rate==16000 and info.subtype=='PCM_16' and pcm.ndim==1 and 3200<=len(pcm)<=160000 and
            len(pcm)==len(resampled)==(len(native)*numerator+denominator-1)//denominator and
            np.isfinite(native).all() and np.isfinite(resampled).all() and np.any(pcm) and np.max(np.abs(native))<1 and
            np.max(np.abs(resampled))<1 and np.min(pcm)>-32768 and np.max(pcm)<32767 and
            np.linalg.norm(resampled.astype(np.float64))/np.sqrt(len(resampled))>=.001 and not r['trimmed'] and not r['normalized'])
        assert bool(good)==r['quality_pass']
        if not good:failures.append(r['order'])
    assert d['quality_pass']==len(rows)-len(failures) and d['reused']==reused
    result=dict(schema='hey-broad-common-negative-independent-audit-v1',engine=a.engine,script_sha256=sha(Path(__file__)),
        original_report_sha256=sha(report),cases=len(rows),reused=reused,files_verified=count,
        independently_qualified=len(rows)-len(failures),failures=failures,
        saved_resampling_and_token_duration_identity=True,
        limits='Saved training-only synthetic process and internal durations, not phonetic truth or trained-model evaluation. PCM format/hash/endpoint checks; no alternate independent PCM rounding.')
    with (RESULTS/f'4090-hey-broad-common-{a.engine}-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
