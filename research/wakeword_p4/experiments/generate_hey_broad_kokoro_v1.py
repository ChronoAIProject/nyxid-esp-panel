"""Full untrimmed Kokoro training matrix; stage lock before any synthesis."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-broad-kokoro-generation-plan-v1.json'
AUX=RESULTS/'4090-hey-broad-aux-material-plan-v1.json'
OUT=BASE/'hey-broad-kokoro-native-v1'
MODEL=BASE/'kokoro-models/kokoro-v1.0.onnx';BANK=MODEL.with_name('voices-v1.0.bin')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['kokoro-onnx','onnxruntime','numpy','scipy','soundfile']}
def inputs(k,r):
    phones=k.tokenizer.phonemize(r['text'],r['lang']);tokens=k.tokenizer.tokenize(phones)
    prefix=k.tokenizer.phonemize(r['prefix_text'],r['lang']);pt=k.tokenizer.tokenize(prefix)
    assert 0<len(tokens)<500 and len(pt)>0 and list(tokens[:len(pt)])==list(pt)
    style=np.asarray(k.get_voice_style(r['conditioning_id'])[len(tokens)],dtype=np.float32)
    return dict(phonemes=phones,prefix_phonemes=prefix,token_ids=[0,*tokens,0],prefix_token_count=len(pt),style_sha256=hashlib.sha256(style.tobytes()).hexdigest()),style

def plan():
    from kokoro_onnx import Kokoro
    aux=json.loads(AUX.read_text());p=Path(aux['schedule']['path']);assert sha(p)==aux['schedule']['sha256']
    assert sha(MODEL)==aux['kokoro']['model_sha256'] and sha(BANK)==aux['kokoro']['bank_sha256']
    k=Kokoro(str(MODEL),str(BANK));requests=[]
    for r in map(json.loads,p.read_text().splitlines()):
        if r['engine']=='kokoro':
            meta,_=inputs(k,r);requests.append(dict(request=r,**meta))
    assert len(requests)==480
    d=dict(schema='hey-broad-kokoro-generation-plan-v1',source_sha256=sha(Path(__file__)),aux_sha256=sha(AUX),
        model_sha256=sha(MODEL),bank_sha256=sha(BANK),versions=versions(),wrapper_sha256=sha(Path(inspect.getfile(Kokoro))),
        requests=requests,prerequisite_audit_sha256=sha(RESULTS/'4090-hey-kokoro-duration-array-audit-v1.json'),
        stage_policy='Supersedes auxiliary subplan procedural wait for complete negative/recipe lock for this synthesis stage only. Exact480 requests already fixed; common negative/training lock still required before training. No other engine or heldout inference.',
        rules='No wrapper trim/split. Direct float32 speed; save complete native/resampled float, original duration/padded tokens/style and PCM16. One attempt per request; retain every failure, no retries/normalization/trimming. Prefix is internal generated duration reference, not acoustic truth.',
        quality='All finite,mono,nonempty,.2..10s,RMS>=.001; native/resampled abs<1; PCM no endpoints; exact ceil resampled length; integer nonnegative duration/token count; sum*600==native samples; exact prefix tokens and mapped boundary.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('480 exact tokens/styles/phones and source/runtime identities locked, no synthesis')

def run():
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    from scipy.signal import resample_poly
    import soundfile as sf
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and versions()==d['versions']
    assert sha(AUX)==d['aux_sha256'] and sha(MODEL)==d['model_sha256'] and sha(BANK)==d['bank_sha256']
    assert sha(Path(inspect.getfile(Kokoro)))==d['wrapper_sha256']
    assert sha(RESULTS/'4090-hey-kokoro-duration-array-audit-v1.json')==d['prerequisite_audit_sha256']
    assert not OUT.exists();OUT.mkdir()
    k=Kokoro(str(MODEL),str(BANK));o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
    sess=ort.InferenceSession(str(MODEL),sess_options=o,providers=['CPUExecutionProvider'])
    assert [x.name for x in sess.get_outputs()]==['waveform','duration']
    records=[]
    with (OUT/'records.jsonl').open('x') as log:
        for i,item in enumerate(d['requests']):
            r=item['request'];meta,style=inputs(k,r)
            assert all(meta[n]==item[n] for n in meta)
            tokens=np.asarray([meta['token_ids']],dtype=np.int64);speed=np.asarray([r['speed']],dtype=np.float32)
            native,duration=sess.run(None,dict(input_ids=tokens,style=style,speed=speed));native=native.reshape(-1)
            resampled=resample_poly(native,2,3)
            arrays=dict(native=native,resampled=resampled,duration=duration,tokens=tokens,style=style,speed=speed);files={}
            for name,a in arrays.items():
                p=OUT/f'{i:04}-{name}.npy';np.save(p,a);files[name]=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)
            p=OUT/f'{i:04}.wav';sf.write(p,resampled,16000,subtype='PCM_16');pcm,rate=sf.read(p,dtype='int16')
            files['wav']=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size)
            dur=duration.reshape(-1);rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
            prefix_native=sum(float(x) for x in dur[:1+meta['prefix_token_count']])*600
            checks=dict(finite=bool(all(np.isfinite(a).all() for a in arrays.values())),
                nonempty=bool(native.size and pcm.size and np.any(pcm)),duration_range=.2<=len(pcm)/16000<=10,
                mono_pcm16=pcm.ndim==1 and rate==16000 and sf.info(p).subtype=='PCM_16',rms=rms>=.001,
                native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),
                pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),
                full_length=len(pcm)==len(resampled)==(len(native)*2+2)//3,
                token_duration=dur.size==tokens.size and bool(np.all(dur>=0) and np.array_equal(dur,np.rint(dur))),
                native_duration=float(np.sum(dur,dtype=np.float64))*600==len(native),
                prefix_boundary=0<prefix_native<=len(native) and prefix_native==int(prefix_native))
            row=dict(order=i,request=r,metadata=meta,files=files,native_samples=len(native),samples=len(pcm),rms=rms,
                generated_prefix_end_native_sample=prefix_native,generated_prefix_end_16k_sample=(int(prefix_native)*2+2)//3,
                pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,quality_pass=all(checks.values()),trimmed=False,normalized=False)
            log.write(json.dumps(row)+'\n');log.flush();records.append(row)
            if (i+1)%20==0:print(json.dumps(dict(completed=i+1,total=480)),flush=True)
    result=dict(schema='hey-broad-kokoro-generation-result-v1',plan_sha256=sha(PLAN),records_sha256=sha(OUT/'records.jsonl'),
        cases=len(records),quality_pass=sum(r['quality_pass'] for r in records),
        failures=[dict(order=r['order'],checks=r['checks']) for r in records if not r['quality_pass']],
        training_launched=False,limits='Synthetic process and internal generated timing only; independent artifact audit still needed; no promotion.')
    with (RESULTS/'4090-hey-broad-kokoro-generation-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
