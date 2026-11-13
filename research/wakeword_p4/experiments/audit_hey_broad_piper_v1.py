"""Independent saved-array audit for full Piper material; never synthesizes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RESULTS=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword/hey-broad-piper-native-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
    pp=RESULTS/'4090-hey-broad-piper-generation-plan-v1.json';rp=RESULTS/'4090-hey-broad-piper-generation-v1.json'
    plan=json.loads(pp.read_text());result=json.loads(rp.read_text())
    assert sha(pp)==result['plan_sha256'] and sha(BASE/'records.jsonl')==result['records_sha256']
    rows=[json.loads(s) for s in (BASE/'records.jsonl').read_text().splitlines()]
    p=Path(plan['schedule']['path']);assert sha(p)==plan['schedule']['sha256']
    requests=[json.loads(s) for s in p.read_text().splitlines()]
    assert len(rows)==len(requests)==16128
    oldrows=[json.loads(s) for s in Path('.local/wakeword/piper-balanced-training-full-v1/manifest.jsonl').read_text().splitlines()]
    failures=[];reused=0;files=0
    receipts=BASE/'independent-array-receipts.jsonl'
    with receipts.open('x') as ledger:
        for i,(r,request) in enumerate(zip(rows,requests)):
            assert r['order']==i and r['request']==request
            meta=plan['text_inventory'][request['text']];assert r['metadata']==meta
            arrays={}
            for name,f in r['files'].items():
                p=Path(f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];files+=1
                if name!='wav':arrays[name]=np.load(p,allow_pickle=False)
            if request['action']=='reuse':
                old=oldrows[request['source_order']]
                assert old['token_ids']==meta['token_ids'] and old['phonemes']==meta['phonemes']
                assert old['files']==request['files'] and old['text']==request['text']
                reused+=1
            native=arrays['native'];resampled=arrays['resampled'];dur=arrays['durations'].ravel()
            assert native.ndim==resampled.ndim==1 and native.dtype==resampled.dtype==np.float32
            assert np.array_equal(resample_poly(native,320,441),resampled)
            assert len(dur)==len(meta['token_ids']) and np.isfinite(dur).all() and np.all(dur>=0)
            assert np.array_equal(dur,dur.astype(np.int64)) and sum(map(int,dur))==len(native)
            end=sum(map(int,dur[:meta['prefix_token_count']]))
            assert end==r['generated_prefix_end_native_sample']
            mapped=(end*16000+22049)//22050;assert mapped==r['generated_prefix_end_16k_sample']
            pcm,rate=sf.read(Path(r['files']['wav']['path']),dtype='int16');info=sf.info(Path(r['files']['wav']['path']))
            assert sha(Path(r['files']['wav']['path']))==r['files']['wav']['sha256']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
            good=(rate==16000 and info.subtype=='PCM_16' and pcm.ndim==1 and 3200<=len(pcm)<=160000 and
                len(pcm)==len(resampled)==(len(native)*320+440)//441 and
                np.isfinite(native).all() and np.isfinite(resampled).all() and np.any(pcm) and
                np.max(np.abs(native))<1 and np.max(np.abs(resampled))<1 and np.min(pcm)>-32768 and np.max(pcm)<32767 and
                np.linalg.norm(resampled.astype(np.float64))/np.sqrt(len(resampled))>=.001 and
                0<end<=len(native) and not r['trimmed'] and not r['normalized'])
            assert bool(good)==r['quality_pass']
            if not good:failures.append(i)
            ledger.write(json.dumps(dict(order=i,qualified=bool(good),prefix_native=end,prefix_16k=mapped))+'\n')
            if (i+1)%1000==0:print(json.dumps(dict(audited=i+1,total=16128)),flush=True)
    assert reused==2016 and result['quality_pass']==16128-len(failures)
    r=dict(schema='hey-broad-piper-independent-audit-v1',script_sha256=sha(Path(__file__)),plan_sha256=sha(pp),original_report_sha256=sha(rp),
        cases=16128,reused_cases=reused,generated_cases=16128-reused,files_verified=files,
        independently_qualified=16128-len(failures),failures=failures,receipt_sha256=sha(receipts),
        full_saved_resample_arrays_reproduced=True,all_token_duration_and_prefix_identities=True,
        limits='Mechanical full saved training material and internal generated reference only; no acoustic truth/recognition/promotion. PCM hash/format/endpoint check, not alternate independent PCM rounding.')
    with (RESULTS/'4090-hey-broad-piper-independent-audit-v1.json').open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);parser.add_argument('--starttime');a=parser.parse_args()
    if a.wait_pid:
        assert a.starttime
        p=Path(f'/proc/{a.wait_pid}/stat')
        while p.exists():
            try:current=p.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if current!=a.starttime:break
            time.sleep(5)
    audit()
if __name__=='__main__':main()
