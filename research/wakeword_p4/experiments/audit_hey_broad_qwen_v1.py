"""Audit original Qwen saved arrays without loading TTS or a wake model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RESULTS=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword/hey-broad-qwen-native-v1')
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
    pp=RESULTS/'4090-hey-broad-qwen-generation-plan-v1.json';rp=RESULTS/'4090-hey-broad-qwen-generation-v1.json'
    plan=json.loads(pp.read_text());report=json.loads(rp.read_text());assert sha(pp)==report['plan_sha256']
    mp=RESULTS/'4090-hey-broad-qwen-standalone-material-plan-v1.json';material=json.loads(mp.read_text());assert sha(mp)==plan['material_sha256']
    records=BASE/'records.jsonl';assert sha(records)==report['records_sha256']
    rows=[json.loads(s) for s in records.read_text().splitlines()];assert len(rows)==len(material['requests'])==360
    files=0;failures=[];counts={};cap=plan['generation']['max_new_tokens']
    for i,(r,request) in enumerate(zip(rows,material['requests'])):
        assert r['order']==i and r['request']==request;arrays={}
        for name,f in r['files'].items():
            path=Path(f['path']);assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'];files+=1
            if name!='wav':
                arr=np.load(path,allow_pickle=False);assert list(arr.shape)==f['shape'] and str(arr.dtype)==f['dtype'];arrays[name]=arr
        native=arrays['native'];resampled=arrays['resampled'];codec=arrays['codec']
        assert native.ndim==resampled.ndim==1 and native.dtype==resampled.dtype==np.float32
        assert np.array_equal(resample_poly(native,2,3),resampled)
        for name in ['codec','text_ids','instruction_ids']:
            assert np.issubdtype(arrays[name].dtype,np.integer) and arrays[name].size>0
        pcm,rate=sf.read(Path(r['files']['wav']['path']),dtype='int16');info=sf.info(Path(r['files']['wav']['path']))
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
        assert r['phonemes'] is None and r['aligned_text_durations'] is None
        if request['role']=='own-target-positive':
            assert r['generated_reference_end_native_sample']==len(native) and r['generated_reference_end_16k_sample']==len(pcm)
        else:assert r['generated_reference_end_native_sample'] is None and r['generated_reference_end_16k_sample'] is None
        good=(r['native_rate']==24000 and rate==16000 and info.subtype=='PCM_16' and pcm.ndim==1 and
            3200<=len(pcm)<=160000 and len(pcm)==len(resampled)==(len(native)*2+2)//3 and
            np.isfinite(native).all() and np.isfinite(resampled).all() and np.any(pcm) and
            np.max(np.abs(native))<1 and np.max(np.abs(resampled))<1 and np.min(pcm)>-32768 and np.max(pcm)<32767 and
            np.linalg.norm(resampled.astype(np.float64))/np.sqrt(len(resampled))>=.001 and
            codec.ndim==2 and 0<codec.shape[0]<cap and not r['trimmed'] and not r['normalized'])
        assert bool(good)==r['quality_pass']
        if not good:failures.append(i)
        key=request.get('target',request['role']);counts[key]=counts.get(key,0)+1
    assert report['quality_pass']==360-len(failures)
    d=dict(schema='hey-broad-qwen-independent-audit-v1',script_sha256=sha(Path(__file__)),original_report_sha256=sha(rp),cases=360,
        files_verified=files,independently_qualified=360-len(failures),failures=failures,roles=counts,
        all_saved_resampling_arrays_reproduced=True,actual_saved_prompt_ids_and_codec_files_verified=True,
        limits='No independent text-tokenizer reproduction or codec decoder rerun. Actual retained prompt/codec array identity, mechanical waveform process and full standalone generated endpoint only; no phonetic truth, independent human identities or model acceptance.')
    with (RESULTS/'4090-hey-broad-qwen-independent-audit-v1.json').open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d),flush=True)
if __name__=='__main__':main()
