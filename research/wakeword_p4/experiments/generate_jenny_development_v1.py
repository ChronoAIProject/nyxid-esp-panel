"""Locked CPU132 native synthesis requests; quality only, no wake inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

RESULTS=Path('research/wakeword_p4/results')
MATERIAL=RESULTS/'4090-jenny-development-material-plan-v1.json'
OBSERVER=RESULTS/'4090-jenny-duration-observer-v1.json'
PLAN=RESULTS/'4090-jenny-development-generation-plan-v1.json'
OUT=Path('.local/wakeword/jenny-development-native-v1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['numpy','scipy','soundfile','onnxruntime']}


def plan():
    m=json.loads(MATERIAL.read_text());o=json.loads(OBSERVER.read_text())
    for p,h in m['files'].items():assert sha(p)==h
    assert sha(MATERIAL)==o['plan_sha256'] and sha(o['path'])==o['sha256']
    paths=[Path(__file__),MATERIAL,OBSERVER,Path(o['path'])]
    paths+=list((Path(ort.__file__).parent/'capi').glob('*.so'))
    d=dict(schema='jenny-development-generation-plan-v1',files={str(p):sha(p) for p in paths},versions=versions(),
        requests=132,provider='CPUExecutionProvider',intra_threads=2,inter_threads=1,
        source_role='Prospective development synthesis only; exclude entire author-direct Jenny family from training and independent acceptance.',
        source_basis='Existing original source/model/config/license receipts plus Mac341ffbf bounded no-use metadata review. Declared corpus differs from training materials. Residual physical-person/shared-pretraining and missing-history uncertainty is retained, not certified away.',
        text_policy='Exactly material132 rows, saved phones/tokens and allocated prefix, native scales; one inference/request, no retries/ASR selection/trim/normalization.',
        seed_policy='ort.set_seed(92726000+order) recorded, not certified to control all stochastic graph nodes. Saved actual outputs identify results.',
        quality_policy=m['quality'],timing_policy=m['timing'],
        authorizes='One CPU native material-generation execution and quality reporting only; no training, wake scoring, threshold selection or acceptance access.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print(json.dumps(dict(requests=132,provider=d['provider'],wake_scoring=False)))


def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();p=json.loads(remote)
    for path,h in p['files'].items():assert sha(path)==h
    assert versions()==p['versions']
    m=json.loads(MATERIAL.read_text());o=json.loads(OBSERVER.read_text())
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(o['path'],sess_options=opts,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider']
    assert [v.name for v in session.get_outputs()]==['output','/Ceil_output_0']
    OUT.mkdir(exist_ok=False);rows=[]
    with (OUT/'records.jsonl').open('x') as log:
        for request in m['requests']:
            index=request['order'];meta=m['texts'][request['text']];tokens=np.asarray([meta['token_ids']],dtype=np.int64)
            seed=92726000+index;ort.set_seed(seed)
            inputs=dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],np.int64),
                scales=np.asarray([request['noise_scale'],request['length_scale'],request['noise_scale_w']],np.float32))
            waveform,duration=session.run(None,inputs)
            assert sum(int(d)>1 for d in waveform.shape)==1 and waveform.dtype==np.float32
            native=waveform.reshape(-1);durations=duration.reshape(-1)
            resampled=resample_poly(native,320,441)
            files={}
            for name,value in [('native',native),('resampled',resampled),('durations',durations),('tokens',tokens),('scales',inputs['scales'])]:
                path=OUT/f'{index:03}-{name}.npy';np.save(path,value,allow_pickle=False)
                files[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
            wav=OUT/f'{index:03}.wav';sf.write(wav,resampled,16000,subtype='PCM_16')
            files['wav']=dict(path=str(wav),bytes=wav.stat().st_size,sha256=sha(wav))
            pcm,rate=sf.read(wav,dtype='int16');prefix=float(durations[:meta['prefix_token_count']].sum(dtype=np.float64))*256
            checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(resampled).all() and np.isfinite(durations).all()),
                format=rate==16000 and pcm.ndim==1 and sf.info(wav).subtype=='PCM_16',
                nonempty=bool(len(pcm)>0 and np.any(pcm)),duration=.2<=len(pcm)/16000<=10,
                rms=bool(np.sqrt(np.mean(resampled.astype(np.float64)**2))>=.001),
                native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),
                pcm_no_endpoint=bool(np.all((pcm!=-32768)&(pcm!=32767))),
                full_length=len(pcm)==len(resampled)==(len(native)*320+440)//441,
                token_durations=len(durations)==tokens.shape[1] and bool(np.all(durations>=0) and np.array_equal(durations,np.rint(durations))),
                duration_sum=float(durations.sum(dtype=np.float64))*256==len(native),prefix=0<prefix<=len(native) and prefix==int(prefix))
            row=dict(request=request,metadata=meta,seed=seed,seed_control_certified=False,files=files,
                native_samples=len(native),samples=len(pcm),native_prefix_end_sample=int(prefix),
                generated_prefix_end_sample=(int(prefix)*320+440)//441,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                checks=checks,quality_pass=all(checks.values()))
            rows.append(row);log.write(json.dumps(row,sort_keys=True)+'\n');log.flush()
            if (index+1)%12==0:print(json.dumps(dict(completed=index+1,total=132,quality_passes=sum(r['quality_pass'] for r in rows))),flush=True)
    result=dict(schema='jenny-development-native-generation-v1',plan_sha256=sha(PLAN),requests=len(rows),
        quality_passes=sum(r['quality_pass'] for r in rows),failed_orders=[r['request']['order'] for r in rows if not r['quality_pass']],
        records_path=str(OUT/'records.jsonl'),records_sha256=sha(OUT/'records.jsonl'),wake_scoring=False,
        limits='Generation process quality only, not human sound/phoneme truth or proof of source independence. Independent saved-array audit and render/history admission remain before wake scoring. No retry/filtering.')
    with (RESULTS/'4090-jenny-development-native-generation-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
