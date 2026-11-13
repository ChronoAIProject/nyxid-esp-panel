"""Pinned full-waveform common negatives, training-only, no wake scoring."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
MATERIAL=RESULTS/'4090-hey-broad-common-negative-plan-v1.json'
PLAN=RESULTS/'4090-hey-broad-common-negative-generation-plan-v1.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['torch','piper-phonemize','kokoro-onnx','onnxruntime','numpy','scipy','soundfile']}
def piper_helpers():
    sys.path.insert(0,str((BASE/'piper').resolve()))
    from generate_samples import generate_audio,get_phonemes
    return generate_audio,get_phonemes

def plan():
    from kokoro_onnx import Kokoro
    from piper_phonemize import phonemize_espeak
    material=json.loads(MATERIAL.read_text());schedule=Path(material['schedule']['path']);assert sha(schedule)==material['schedule']['sha256']
    pp=RESULTS/'4090-hey-broad-piper-generation-plan-v1.json';pd=json.loads(pp.read_text())
    kp=RESULTS/'4090-hey-broad-kokoro-generation-plan-v1.json';kd=json.loads(kp.read_text())
    config=json.loads((BASE/'piper/models/en_US-libritts_r-medium.pt.json').read_text());_,get=piper_helpers()
    k=Kokoro(str(BASE/'kokoro-models/kokoro-v1.0.onnx'),str(BASE/'kokoro-models/voices-v1.0.bin'))
    texts={}
    for text in material['texts']:
        phones=phonemize_espeak(text,config['espeak']['voice']);assert len(phones)==1 and all(p in config['phoneme_id_map'] for p in phones[0])
        tokens,clip=get(config['espeak']['voice'],config,text);assert clip is None
        texts['piper|'+text]=dict(phonemes=phones[0],token_ids=tokens)
        for lang in ['en-us','en-gb']:
            phones=k.tokenizer.phonemize(text,lang);tokens=k.tokenizer.tokenize(phones);assert 0<len(tokens)<500
            texts['kokoro|'+lang+'|'+text]=dict(phonemes=phones,token_ids=[0,*tokens,0])
    d=dict(schema='hey-broad-common-negative-generation-plan-v1',source_sha256=sha(Path(__file__)),material_sha256=sha(MATERIAL),
        schedule=material['schedule'],versions=versions(),source_plans={str(pp):sha(pp),str(kp):sha(kp)},
        piper_model_sha256=pd['model_sha256'],piper_config_sha256=pd['config_sha256'],piper_source_files=pd['source_files'],
        kokoro_model_sha256=kd['model_sha256'],kokoro_bank_sha256=kd['bank_sha256'],wrapper_sha256=sha(Path(inspect.getfile(Kokoro))),
        text_inventory=texts,allowed_engines=['piper','kokoro'],
        rules='Exact existing common schedule, full native outputs/token durations/float resample/PCM. Same mechanical checks as positive stage; .2..10seconds. Reuse original artifacts unchanged. No retries/trim/normalization, preserve all failures. Negatives have no invented wake-prefix boundary.',
        stage_policy='Generation only; does not admit model training or open heldout. Qwen216 remains separate pinned stage.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('Piper/Kokoro common negative tokens, sources and runtime fixed')

def run(engine,wait_pid,starttime):
    if wait_pid:
        assert starttime
        proc=Path(f'/proc/{wait_pid}/stat')
        while proc.exists():
            try:s=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if s!=starttime:break
            time.sleep(5)
        original=json.loads((RESULTS/'4090-hey-broad-piper-generation-v1.json').read_text());assert original['cases']==16128
    import soundfile as sf
    from scipy.signal import resample_poly
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and sha(MATERIAL)==d['material_sha256'] and versions()==d['versions']
    for p,h in d['source_plans'].items():assert sha(Path(p))==h
    p=Path(d['schedule']['path']);assert sha(p)==d['schedule']['sha256']
    rows=[r for r in map(json.loads,p.read_text().splitlines()) if r['engine']==engine]
    out=BASE/f'hey-broad-common-{engine}-native-v1';assert not out.exists();out.mkdir()
    if engine=='piper':
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(4)
        modelpath=BASE/'piper/models/en_US-libritts_r-medium.pt'
        assert sha(modelpath)==d['piper_model_sha256'] and sha(Path(str(modelpath)+'.json'))==d['piper_config_sha256']
        for path,h in d['piper_source_files'].items():assert sha(Path(path))==h
        fn,_=piper_helpers();model=torch.load(modelpath,map_location='cpu',weights_only=False).eval().cuda()
    else:
        from kokoro_onnx import Kokoro
        import onnxruntime as ort
        assert sha(Path(inspect.getfile(Kokoro)))==d['wrapper_sha256']
        mp=BASE/'kokoro-models/kokoro-v1.0.onnx';bp=BASE/'kokoro-models/voices-v1.0.bin'
        assert sha(mp)==d['kokoro_model_sha256'] and sha(bp)==d['kokoro_bank_sha256']
        k=Kokoro(str(mp),str(bp));o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
        sess=ort.InferenceSession(str(mp),sess_options=o,providers=['CPUExecutionProvider'])
    failures=[];reused=0
    with (out/'records.jsonl').open('x') as log:
        for i,r in enumerate(rows):
            key=engine+'|'+(r['lang']+'|' if engine=='kokoro' else '')+r['text'];meta=d['text_inventory'][key]
            files={}
            if r['action']=='reuse':
                old=BASE/'piper-balanced-training-full-v1'
                for name,f in r['files'].items():
                    p=old/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['size_bytes'];files[name]=dict(path=str(p),sha256=f['sha256'],bytes=f['size_bytes'])
                native=np.load(Path(files['native']['path']),allow_pickle=False);resampled=np.load(Path(files['resampled']['path']),allow_pickle=False)
                dur=np.load(Path(files['durations']['path']),allow_pickle=False).ravel();wav=Path(files['wav']['path']);reused+=1
            else:
                if engine=='piper':
                    torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
                    with torch.inference_mode():audio,duration=fn(model,torch.LongTensor([r['conditioning_id']]),torch.LongTensor([r['conditioning_id']]),[meta['token_ids']],.5,.333,.333,r['scale'],None)
                    native=audio.detach().cpu().numpy()[0,0];dur=duration.detach().cpu().numpy()[0,0];resampled=resample_poly(native,320,441)
                    arrays=dict(native=native,resampled=resampled,durations=dur)
                else:
                    tokens=np.asarray([meta['token_ids']],dtype=np.int64);style=np.asarray(k.get_voice_style(r['conditioning_id'])[len(meta['token_ids'])-2],dtype=np.float32);speed=np.asarray([r['scale']],dtype=np.float32)
                    audio,duration=sess.run(None,dict(input_ids=tokens,style=style,speed=speed));native=audio.ravel();dur=duration.ravel();resampled=resample_poly(native,2,3)
                    arrays=dict(native=native,resampled=resampled,durations=dur,tokens=tokens,style=style,speed=speed)
                for name,a in arrays.items():
                    p=out/f'{i:05}-{name}.npy';np.save(p,a);files[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
                wav=out/f'{i:05}.wav';sf.write(wav,resampled,16000,subtype='PCM_16');files['wav']=dict(path=str(wav),sha256=sha(wav),bytes=wav.stat().st_size)
            pcm,rate=sf.read(wav,dtype='int16');rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
            divisor,numerator=(441,320) if engine=='piper' else (3,2)
            checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(resampled).all() and np.isfinite(dur).all()),
                format=pcm.ndim==1 and rate==16000 and sf.info(wav).subtype=='PCM_16',nonempty=bool(len(pcm) and np.any(pcm)),
                duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(native)<1)),
                resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),
                full_length=len(pcm)==len(resampled)==(len(native)*numerator+divisor-1)//divisor,
                token_duration=len(dur)==len(meta['token_ids']) and bool(np.all(dur>=0) and np.array_equal(dur,np.rint(dur))),
                native_duration=float(np.sum(dur,dtype=np.float64))*(1 if engine=='piper' else 600)==len(native))
            good=all(checks.values())
            if not good:failures.append(dict(order=r['order'],checks=checks))
            log.write(json.dumps(dict(order=r['order'],request=r,metadata=meta,files=files,native_samples=len(native),samples=len(pcm),rms=rms,
                pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,quality_pass=good,trimmed=False,normalized=False))+'\n');log.flush()
            if (i+1)%100==0:print(json.dumps(dict(engine=engine,completed=i+1,total=len(rows))),flush=True)
    result=dict(schema='hey-broad-common-negative-generation-result-v1',engine=engine,plan_sha256=sha(PLAN),records_sha256=sha(out/'records.jsonl'),
        cases=len(rows),reused=reused,quality_pass=len(rows)-len(failures),failures=failures,training_launched=False)
    with (RESULTS/f'4090-hey-broad-common-{engine}-generation-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('--engine',choices=['piper','kokoro']);p.add_argument('--wait-pid',type=int);p.add_argument('--starttime');a=p.parse_args()
    if a.action=='plan':plan()
    else:
        assert a.engine;run(a.engine,a.wait_pid,a.starttime)
