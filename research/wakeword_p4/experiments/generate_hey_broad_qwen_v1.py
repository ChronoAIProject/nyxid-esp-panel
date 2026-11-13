"""Original Qwen standalone synthesis with actual prompt/codec retention."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import time
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
MATERIAL=RESULTS/'4090-hey-broad-qwen-standalone-material-plan-v1.json'
PLAN=RESULTS/'4090-hey-broad-qwen-generation-plan-v1.json'
OUT=BASE/'hey-broad-qwen-native-v1';MODEL=BASE/'qwen3-design'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['torch','torchaudio','qwen-tts','transformers','numpy','scipy','soundfile']}
def plan():
    from qwen_tts import Qwen3TTSModel
    material=json.loads(MATERIAL.read_text());origin=material['original_qwen']
    for p,h in origin['model_files'].items():assert sha(MODEL/p)==h
    pkg=Path(inspect.getfile(Qwen3TTSModel)).parents[1]
    d=dict(schema='hey-broad-qwen-generation-plan-v1',source_sha256=sha(Path(__file__)),material_sha256=sha(MATERIAL),
        versions=versions(),package_sources={str(p):sha(p) for p in sorted(pkg.rglob('*.py'))},
        model_files=origin['model_files'],generation=origin['generation'],requests=360,
        policy='One attempt per locked request, bfloat16/SDPA/English/CUDA, original max96 codec tokens. Save actual input/instruction IDs and returned codec codes through observational same-instance generate wrapper; forward original return unchanged. No retries/trim/normalization/ASR filtering.',
        quality='Native float32/finite/mono, .2..10seconds, RMS>=.001, native/resampled abs<1, PCM no endpoint, complete ceil2/3 resample length at native24k, nonempty codec matrix with sequence length<max_new_tokens. Preserve every failure.',
        timing='Phonemes and aligned text durations unavailable. Bare positive generated endpoint is full native waveform end. No command timing or codec-to-text alignment inferred.',
        stage_policy='Synthesis only; no training, heldout or promotion. Can wait exact original Piper negative PID/start before using GPU.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('360 Qwen requests/source/runtime/native-output retention locked, no inference')

def run(wait_pid,starttime):
    if wait_pid:
        assert starttime;proc=Path(f'/proc/{wait_pid}/stat')
        while proc.exists():
            try:s=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if s!=starttime:break
            time.sleep(5)
        assert json.loads((RESULTS/'4090-hey-broad-common-piper-generation-v1.json').read_text())['cases']==18144
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and sha(MATERIAL)==d['material_sha256'] and versions()==d['versions']
    for p,h in d['package_sources'].items():assert sha(Path(p))==h
    for p,h in d['model_files'].items():assert sha(MODEL/p)==h
    material=json.loads(MATERIAL.read_text());requests=material['requests'];assert len(requests)==d['requests']==360
    assert torch.cuda.is_available() and not OUT.exists();OUT.mkdir()
    torch.set_num_threads(4)
    model=Qwen3TTSModel.from_pretrained(str(MODEL.resolve()),device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    original=model.model.generate;captured={}
    def observe(*args,**kwargs):
        assert not args
        for name in ['input_ids','instruct_ids']:
            captured[name]=[None if t is None else t.detach().cpu().numpy().copy() for t in kwargs[name]]
        result=original(*args,**kwargs)
        assert len(result[0])==1
        captured['codec']=result[0][0].detach().cpu().numpy().copy()
        return result
    model.model.generate=observe
    failures=[]
    with (OUT/'records.jsonl').open('x') as log:
        for i,r in enumerate(requests):
            captured.clear();torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            with torch.inference_mode():wavs,rate=model.generate_voice_design(text=r['text'],instruct=r['description'],language='English',**d['generation'])
            assert len(wavs)==1 and rate==24000
            native=np.asarray(wavs[0]);assert native.dtype==np.float32 and native.ndim==1
            assert len(captured['input_ids'])==len(captured['instruct_ids'])==1
            resampled=resample_poly(native,2,3)
            arrays=dict(native=native,resampled=resampled,codec=captured['codec'],text_ids=captured['input_ids'][0],instruction_ids=captured['instruct_ids'][0])
            assert all(a is not None for a in arrays.values());files={}
            for name,a in arrays.items():
                p=OUT/f'{i:04}-{name}.npy';np.save(p,a);files[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,shape=list(a.shape),dtype=str(a.dtype))
            wav=OUT/f'{i:04}.wav';sf.write(wav,resampled,16000,subtype='PCM_16');pcm,sr=sf.read(wav,dtype='int16')
            files['wav']=dict(path=str(wav),sha256=sha(wav),bytes=wav.stat().st_size)
            rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
            codec=captured['codec']
            checks=dict(finite=bool(all(np.isfinite(a).all() for a in arrays.values())),format=sr==16000 and pcm.ndim==1 and sf.info(wav).subtype=='PCM_16',
                nonempty=bool(len(native) and len(pcm) and np.any(pcm)),duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,
                native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),
                pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),full_length=len(pcm)==len(resampled)==(len(native)*2+2)//3,
                codec_not_capped=codec.ndim==2 and 0<codec.shape[0]<d['generation']['max_new_tokens'])
            good=all(checks.values())
            if not good:failures.append(dict(order=i,checks=checks))
            row=dict(order=i,request=r,files=files,native_samples=len(native),samples=len(pcm),native_rate=rate,rms=rms,
                pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,quality_pass=good,
                generated_reference_end_native_sample=len(native) if r['role']=='own-target-positive' else None,
                generated_reference_end_16k_sample=len(pcm) if r['role']=='own-target-positive' else None,
                phonemes=None,aligned_text_durations=None,trimmed=False,normalized=False)
            log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(dict(completed=i+1,total=360)),flush=True)
    result=dict(schema='hey-broad-qwen-generation-result-v1',plan_sha256=sha(PLAN),records_sha256=sha(OUT/'records.jsonl'),cases=360,
        quality_pass=360-len(failures),failures=failures,gpu=torch.cuda.get_device_name(0),training_launched=False,
        limits='Mechanical process and full bare clip generated endpoint only, not phonetic truth/independent people or model acceptance; independent artifact audit required.')
    with (RESULTS/'4090-hey-broad-qwen-generation-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('--wait-pid',type=int);p.add_argument('--starttime');a=p.parse_args()
    plan() if a.action=='plan' else run(a.wait_pid,a.starttime)
