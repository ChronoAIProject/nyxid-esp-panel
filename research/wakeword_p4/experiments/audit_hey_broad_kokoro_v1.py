"""Read-only full saved material audit; no TTS/model import or inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASE=Path('.local/wakeword/hey-broad-kokoro-native-v1')
RESULTS=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    planpath=RESULTS/'4090-hey-broad-kokoro-generation-plan-v1.json'
    reportpath=RESULTS/'4090-hey-broad-kokoro-generation-v1.json'
    d=json.loads(reportpath.read_text());plan=json.loads(planpath.read_text())
    assert sha(planpath)==d['plan_sha256']
    assert sha(BASE/'records.jsonl')==d['records_sha256']
    rows=[json.loads(s) for s in (BASE/'records.jsonl').read_text().splitlines()]
    assert len(rows)==len(plan['requests'])==480
    failures=[];boundaries=[];all_files=0
    for i,(r,p) in enumerate(zip(rows,plan['requests'])):
        assert r['order']==i and r['request']==p['request']
        assert r['metadata']=={k:v for k,v in p.items() if k!='request'}
        arrays={}
        for name,meta in r['files'].items():
            path=BASE/meta['path'];assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'];all_files+=1
            if name!='wav':arrays[name]=np.load(path,allow_pickle=False)
        native=arrays['native'];resampled=arrays['resampled'];dur=arrays['duration'].ravel()
        tokens=arrays['tokens'].ravel();style=arrays['style'];speed=arrays['speed']
        assert native.ndim==resampled.ndim==1 and native.dtype==resampled.dtype==np.float32
        assert np.array_equal(resample_poly(native,2,3),resampled)
        assert list(tokens)==p['token_ids'] and dur.size==tokens.size
        assert hashlib.sha256(style.tobytes()).hexdigest()==p['style_sha256']
        assert speed.dtype==np.float32 and speed.tolist()==np.asarray([p['request']['speed']],dtype=np.float32).tolist()
        assert np.isfinite(dur).all() and np.all(dur>=0) and np.array_equal(dur,dur.astype(np.int64))
        count=sum(map(int,dur))*600;assert count==len(native)
        prefix=sum(map(int,dur[:1+p['prefix_token_count']]))*600
        mapped=(prefix*16000+23999)//24000
        assert prefix==r['generated_prefix_end_native_sample'] and mapped==r['generated_prefix_end_16k_sample']
        pcm,rate=sf.read(BASE/r['files']['wav']['path'],dtype='int16');info=sf.info(BASE/r['files']['wav']['path'])
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
        # Independent quality recomputation; no use of the claimed per-check booleans.
        good=(rate==16000 and info.subtype=='PCM_16' and pcm.ndim==1 and
              3200<=len(pcm)<=160000 and len(pcm)==len(resampled)==(len(native)*2+2)//3 and
              np.isfinite(native).all() and np.isfinite(resampled).all() and np.any(pcm) and
              np.max(np.abs(native))<1 and np.max(np.abs(resampled))<1 and
              np.min(pcm)>-32768 and np.max(pcm)<32767 and
              np.linalg.norm(resampled.astype(np.float64))/np.sqrt(len(resampled))>=.001 and
              0<prefix<=len(native) and not r['trimmed'] and not r['normalized'])
        assert bool(good)==r['quality_pass']
        if not good:failures.append(i)
        boundaries.append(dict(order=i,prefix_native_samples=prefix,prefix_16k_samples=mapped))
    assert d['quality_pass']==480-len(failures)
    result=dict(schema='hey-broad-kokoro-independent-audit-v1',script_sha256=sha(Path(__file__)),
        original_report_sha256=sha(reportpath),plan_sha256=sha(planpath),cases=480,files_verified=all_files,
        independently_qualified=480-len(failures),failures=failures,all_native_duration_identities=True,
        all_saved_resampled_arrays_reproduced=True,boundaries=boundaries,
        limits='Saved full training-only outputs and internal token allocation reference; not phonetic truth, physical-speaker independence, quality of other engines, trained-model performance or promotion. PCM SHA/format/endpoints checked; no alternate independent PCM rounding implementation.')
    with (RESULTS/'4090-hey-broad-kokoro-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='boundaries'}))
if __name__=='__main__':main()
