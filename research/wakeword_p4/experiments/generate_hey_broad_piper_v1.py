"""Frozen broad training-only Piper synthesis, original reused artifacts unchanged."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

BASE=Path('.local/wakeword');SOURCE=BASE/'piper';RESULTS=Path('research/wakeword_p4/results')
SUBPLAN=RESULTS/'4090-hey-broad-piper-material-plan-v1.json'
PLAN=RESULTS/'4090-hey-broad-piper-generation-plan-v1.json'
OUT=BASE/'hey-broad-piper-native-v1';OLD=BASE/'piper-balanced-training-full-v1'
MODEL=SOURCE/'models/en_US-libritts_r-medium.pt';CONFIG=Path(str(MODEL)+'.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['torch','piper-phonemize','numpy','scipy','soundfile']}
def helpers():
    sys.path.insert(0,str(SOURCE.resolve()))
    from generate_samples import generate_audio,get_phonemes
    return generate_audio,get_phonemes

def phonetics(text,config):
    from piper_phonemize import phonemize_espeak
    _,get=helpers();sentences=phonemize_espeak(text,config['espeak']['voice'])
    assert len(sentences)==1
    phones=sentences[0];ids=config['phoneme_id_map'];assert all(p in ids for p in phones)
    tokens,clip=get(config['espeak']['voice'],config,text);assert clip is None
    marker=',' if ',' in text else '.';cut=phones.index(marker)
    count=len(ids['^'])+len(ids['_'])+sum(len(ids[p])+len(ids['_']) for p in phones[:cut])
    assert 0<count<len(tokens)
    return dict(phonemes=phones,token_ids=tokens,prefix_token_count=count,prefix_marker=marker)

def plan():
    sub=json.loads(SUBPLAN.read_text());path=Path(sub['schedule']['path']);assert sha(path)==sub['schedule']['sha256']
    assert sha(MODEL)==sub['model_sha256'] and sha(CONFIG)==sub['config_sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()];config=json.loads(CONFIG.read_text())
    texts={r['text']:phonetics(r['text'],config) for r in rows}
    d=dict(schema='hey-broad-piper-generation-plan-v1',source_sha256=sha(Path(__file__)),subplan_sha256=sha(SUBPLAN),
        schedule=sub['schedule'],model_sha256=sha(MODEL),config_sha256=sha(CONFIG),versions=versions(),
        piper_revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p):sha(p) for p in sorted(SOURCE.rglob('*.py')) if '.git' not in p.parts},
        text_inventory=texts,generated_cells=14112,reused_cells=2016,
        rules='Single original request per cell; direct generate_audio with same original conditioning ID, per-cell seed from schedule, complete native float/durations/resampled float/PCM saved. No clipping, trimming, normalization or retries. Keep all failures.',
        stage_policy='Generation-only registration supersedes subplan wait for complete all-engine negative/recipe lock solely for this fixed material stage. Training remains disabled pending complete recipe/data admission.',
        quality='finite, mono16kPCM16, duration.2..10s, RMS>=.001, no float fullscale/PCM endpoint, full ceil resample length, integer nonnegative duration/token count, duration sum equals native length, full prefix within output.',
        prefix_reference='Before comma or period, includes initial/inter-token blank allocations; not acoustic phonetic truth.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('Eight exact text/token/prefix maps and 16128cell source/runtime identities fixed')

def run():
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    assert sha(Path(__file__))==d['source_sha256'] and sha(SUBPLAN)==d['subplan_sha256'] and versions()==d['versions']
    assert sha(MODEL)==d['model_sha256'] and sha(CONFIG)==d['config_sha256']
    for path,h in d['source_files'].items():assert sha(Path(path))==h
    path=Path(d['schedule']['path']);assert sha(path)==d['schedule']['sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==16128
    assert torch.cuda.is_available() and not OUT.exists();OUT.mkdir()
    torch.set_num_threads(4);audio_fn,_=helpers()
    model=torch.load(MODEL,map_location='cpu',weights_only=False).eval().cuda()
    oldrows=[json.loads(s) for s in (OLD/'manifest.jsonl').read_text().splitlines()]
    records=[]
    with (OUT/'records.jsonl').open('x') as log,torch.inference_mode():
        for i,r in enumerate(rows):
            meta=d['text_inventory'][r['text']]
            if r['action']=='reuse':
                old=oldrows[r['source_order']];assert old['text']==r['text'] and old['files']==r['files']
                files={}
                for name,f in old['files'].items():
                    p=OLD/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['size_bytes']
                    files[name]=dict(path=str(p),sha256=f['sha256'],bytes=f['size_bytes'])
                native=np.load(OLD/old['files']['native']['path'],allow_pickle=False)
                resampled=np.load(OLD/old['files']['resampled']['path'],allow_pickle=False)
                durations=np.load(OLD/old['files']['durations']['path'],allow_pickle=False).reshape(-1)
                wav=OLD/old['files']['wav']['path']
            else:
                torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
                audio,duration=audio_fn(model,torch.LongTensor([r['speaker_id']]),torch.LongTensor([r['speaker_id']]),
                    [meta['token_ids']],r['slerp_weight'],r['noise_scale'],r['noise_scale_w'],r['length_scale'],None)
                native=audio.detach().cpu().numpy()[0,0];durations=duration.detach().cpu().numpy()[0,0]
                resampled=resample_poly(native,320,441);files={}
                for name,a in [('native',native),('resampled',resampled),('durations',durations)]:
                    p=OUT/f'{i:05}-{name}.npy';np.save(p,a);files[name]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
                wav=OUT/f'{i:05}.wav';sf.write(wav,resampled,16000,subtype='PCM_16')
                files['wav']=dict(path=str(wav),sha256=sha(wav),bytes=wav.stat().st_size)
            pcm,rate=sf.read(wav,dtype='int16');rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
            prefix=float(np.sum(durations[:meta['prefix_token_count']],dtype=np.float64))
            checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(resampled).all() and np.isfinite(durations).all()),
                format=pcm.ndim==1 and rate==16000 and sf.info(wav).subtype=='PCM_16',
                nonempty=bool(len(native) and len(pcm) and np.any(pcm)),duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,
                native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),
                pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),
                full_length=len(pcm)==len(resampled)==(len(native)*320+440)//441,
                token_duration=len(durations)==len(meta['token_ids']) and bool(np.all(durations>=0) and np.array_equal(durations,np.rint(durations))),
                native_duration=float(np.sum(durations,dtype=np.float64))==len(native),prefix=0<prefix<=len(native) and prefix==int(prefix))
            row=dict(order=i,request=r,metadata=meta,files=files,native_samples=len(native),samples=len(pcm),
                generated_prefix_end_native_sample=prefix,generated_prefix_end_16k_sample=(int(prefix)*16000+22049)//22050,
                rms=rms,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,
                quality_pass=all(checks.values()),trimmed=False,normalized=False)
            log.write(json.dumps(row)+'\n');log.flush();records.append(row)
            if (i+1)%100==0:print(json.dumps(dict(completed=i+1,total=len(rows))),flush=True)
    result=dict(schema='hey-broad-piper-generation-result-v1',plan_sha256=sha(PLAN),records_sha256=sha(OUT/'records.jsonl'),cases=len(records),
        quality_pass=sum(r['quality_pass'] for r in records),failures=[dict(order=r['order'],checks=r['checks']) for r in records if not r['quality_pass']],
        gpu=torch.cuda.get_device_name(0),training_launched=False,limits='Mechanical synthetic process/internal duration only; independent audit required, no model promotion.')
    with (RESULTS/'4090-hey-broad-piper-generation-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
