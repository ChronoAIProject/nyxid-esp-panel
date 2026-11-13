"""Registered training-only raw Kokoro output/timing probe; no wake-model inference."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np

RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-kokoro-duration-probe-plan-v1.json'
OUT=Path('.local/wakeword/hey-kokoro-duration-probe-v1')
MODEL=Path('.local/wakeword/kokoro-models/kokoro-v1.0.onnx')
BANK=MODEL.with_name('voices-v1.0.bin')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ('kokoro-onnx','onnxruntime','numpy','soundfile')}

def plan():
    from kokoro_onnx import Kokoro
    requests=[]
    for voice in ('af_bella','bm_lewis'):
        for prefix in ('Hey Willow','Hey Nicks'):
            for suffix in ('.',', turn on the light.'):
                for speed in (.9,1.,1.1):
                    requests.append(dict(voice=voice,text=prefix+suffix,prefix=prefix,speed=speed,lang='en-gb' if voice.startswith('b') else 'en-us'))
    d=dict(schema='hey-kokoro-duration-probe-plan-v1',source_sha256=sha(Path(__file__)),
           model_sha256=sha(MODEL),bank_sha256=sha(BANK),versions=versions(),
           installed_wrapper_sha256=sha(Path(inspect.getfile(Kokoro))),requests=requests,
           policy='Exactly24 training-only full untrimmed single-pass raw outputs; no retries/normalization/filtering. Direct actual float32 speed feed, preserve padded token IDs/style/waveform/duration. No wake scoring.',
           timing='Record duration shape/sum and waveform size; infer no phonetic truth. Prefix mapping only if tokenization prefix is exact, otherwise unavailable.',
           heldout_used=False)
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')

def run():
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes()
    d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and versions()==d['versions']
    assert sha(MODEL)==d['model_sha256'] and sha(BANK)==d['bank_sha256']
    assert sha(Path(inspect.getfile(Kokoro)))==d['installed_wrapper_sha256']
    assert not OUT.exists();OUT.mkdir()
    k=Kokoro(str(MODEL),str(BANK));options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider'])
    assert [x.name for x in session.get_outputs()]==['waveform','duration']
    records=[]
    with (OUT/'records.jsonl').open('x') as log:
        for i,r in enumerate(d['requests']):
            phones=k.tokenizer.phonemize(r['text'],r['lang']);tokens=k.tokenizer.tokenize(phones)
            assert len(tokens)<500 and tokens
            padded=np.asarray([[0,*tokens,0]],dtype=np.int64)
            style=np.asarray(k.get_voice_style(r['voice'])[len(tokens)],dtype=np.float32)
            speed=np.asarray([r['speed']],dtype=np.float32)
            waveform,duration=session.run(None,dict(input_ids=padded,style=style,speed=speed))
            arrays=dict(tokens=padded,style=style,speed=speed,waveform=waveform,duration=duration)
            files={}
            for name,a in arrays.items():
                p=OUT/f'{i:02}-{name}.npy';np.save(p,a);files[name]=dict(path=p.name,sha256=sha(p),shape=list(a.shape),dtype=str(a.dtype))
            prefixphones=k.tokenizer.phonemize(r['prefix'],r['lang']);prefixtokens=k.tokenizer.tokenize(prefixphones)
            exact=list(tokens[:len(prefixtokens)])==list(prefixtokens)
            row=dict(order=i,request=r,phonemes=phones,prefix_phonemes=prefixphones,files=files,
                waveform_samples=int(waveform.size),duration_sum=float(np.sum(duration,dtype=np.float64)),
                samples_per_duration_unit=float(waveform.size/np.sum(duration)) if np.sum(duration)>0 else None,
                token_count=int(padded.size),duration_count=int(duration.size),prefix_tokens_exact=exact,
                prefix_token_count=len(prefixtokens) if exact else None,
                finite=bool(np.isfinite(waveform).all() and np.isfinite(duration).all()),
                peak=float(np.max(np.abs(waveform))),trimmed=False,normalized=False)
            log.write(json.dumps(row)+'\n');log.flush();records.append(row)
            print(json.dumps(dict(completed=i+1,total=len(d['requests']))),flush=True)
    summary=dict(schema='hey-kokoro-duration-probe-result-v1',plan_sha256=sha(PLAN),records_sha256=sha(OUT/'records.jsonl'),
        cases=len(records),finite=sum(r['finite'] for r in records),
        exact_prefix_cases=sum(r['prefix_tokens_exact'] for r in records),
        duration_matches_token_count=all(r['duration_count']==r['token_count'] for r in records),
        sample_ratios=sorted({r['samples_per_duration_unit'] for r in records}),
        fullscale_cases=[r['order'] for r in records if r['peak']>=1],
        records=records,limits='Raw model/internal duration evidence only, not phonetic truth, mechanical certification of all materials or training admission.')
    with (RESULTS/'4090-hey-kokoro-duration-probe-v1.json').open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
