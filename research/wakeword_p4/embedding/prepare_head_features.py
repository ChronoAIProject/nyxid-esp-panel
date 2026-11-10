"""Prepare private training/validation histories for a frozen INT8 backbone.

All source audio and generated feature arrays remain outside the repository.
The source index or explicit jobs must retain disjoint training/validation roles.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve,lfilter
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from native_mel import NativeMel

model = None
frontend = None


def initialize(model_path, window_path, filter_path):
    global model, frontend
    model = Path(model_path)
    frontend = NativeMel(np.load(window_path), np.load(filter_path))


def extract(pcm):
    # Keep reference kernels: XNNPACK changes bytes for this stateful graph.
    detector=Interpreter(model_path=str(model),num_threads=1,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    detector.allocate_tensors()
    di,do=detector.get_input_details()[0],detector.get_output_details()[0]
    scale,zero=di['quantization']
    pcm=np.clip(np.rint(pcm),-32768,32767).astype(np.int16)
    pcm=np.pad(pcm,(480,(-len(pcm))%1280))
    result=[]
    for at in range(0,len(pcm)-480,1280):
        features=frontend.block(pcm[at:at+1760])
        assert features.shape==(8,32)
        x=np.clip(np.rint(features/scale)+zero,-128,127).astype(np.int8)
        detector.set_tensor(di['index'],x[None,:,:,None]);detector.invoke()
        result.append(detector.get_tensor(do['index']).reshape(96))
    return np.asarray(result,np.int8)


def worker(job):
    path=Path(job['source']);data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=job['source_sha256']:
        raise ValueError('Source checksum differs')
    pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1
    source_samples=len(pcm)
    if job['kind']=='tts':
        rng=np.random.default_rng(int(job['id'][:8],16))
        lead=38400 #2.4sec covers16embedding frames +76mel context.
        signal=np.pad(pcm.astype(np.float32),(lead,16000))
        if job['variant']:
            ir=np.zeros(2400,np.float32);ir[0]=1
            taps=rng.integers(80,len(ir),24)
            ir[taps]+=rng.normal(0,.2,len(taps))*np.exp(-taps/rng.uniform(500,1400))
            signal=fftconvolve(signal,ir)[:len(signal)]
            level=float(rng.uniform(.04,.20))*32767
            speech_rms=max(655,float(np.sqrt(np.mean(pcm.astype(np.float64)**2))))
            signal*=level/speech_rms
            noise=lfilter([1],[1,-float(rng.uniform(.5,.96))],rng.normal(size=len(signal)))
            noise/=max(1e-9,float(np.sqrt(np.mean(noise**2))))
            signal+=noise*level*10**(-float(rng.uniform(8,30))/20)
        pcm=signal
        keyword_end=(lead+job['keyword_end'])/16000 if job['label'] else None
    else:
        keyword_end=None
    embeddings=extract(pcm)
    output=Path(job['output']);output.parent.mkdir(parents=True,exist_ok=True)
    np.save(output,embeddings)
    row={k:job[k] for k in ['id','split','kind','label','source_sha256','variant','engine']}
    row.update({'path':str(output),'frames':len(embeddings),'seconds':len(pcm)/16000,
                'source_seconds':source_samples/16000,'keyword_end':keyword_end,
                'features_sha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int)
    parser.add_argument('--jobs',type=Path)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--expected-model-sha256',required=True)
    parser.add_argument('--window',type=Path,required=True)
    parser.add_argument('--mel-filter',type=Path,required=True)
    parser.add_argument('--source-features',type=Path)
    args=parser.parse_args()
    if args.workers<=0 or (args.limit is not None and args.limit<=0):parser.error('workers/limit must be positive')
    if not args.jobs and not args.source_features:parser.error('--source-features or --jobs is required')
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=args.expected_model_sha256:
        raise ValueError('Frozen backbone identity differs')
    initialize(args.model,args.window,args.mel_filter)
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.jobs:
        jobs=json.loads(args.jobs.read_text())
        for job in jobs:job['output']=str(args.output/job['split']/job['kind']/(job['id']+'.npy'))
    else:
        records=json.loads((args.source_features/'features.json').read_text())
        sources=[r for r in records if r['variant']==0 and r['split'] in ('training','validation')]
        assert len({r['path'] for r in sources})==len(sources)
        jobs=[]
        for row in sources:
            for variant in (0,1):
                identity=hashlib.sha256((row['path']+':'+str(variant)).encode()).hexdigest()
                jobs.append({'id':identity,'kind':'tts','split':row['split'],'label':row['label'],
                    'source':str(args.source_features/'dataset'/row['path']),
                    'source_sha256':row['sha256'],'keyword_end':row.get('keyword_end'),
                    'variant':variant,'engine':row['engine'],
                    'output':str(args.output/row['split']/'tts'/(identity+'.npy'))})
    if any(job['split'] not in ('training','validation') for job in jobs):
        raise ValueError('Testing jobs are not accepted by this extractor')
    if len({job['id'] for job in jobs})!=len(jobs):raise ValueError('Duplicate source/variant job ID')
    if args.limit:jobs=jobs[:args.limit]
    (args.output/'jobs-private.json').write_text(json.dumps(jobs)+'\n')
    (args.output/'identity.json').write_text(json.dumps({'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
        'window_sha256':hashlib.sha256(args.window.read_bytes()).hexdigest(),
        'mel_filter_sha256':hashlib.sha256(args.mel_filter.read_bytes()).hexdigest(),
        'source_feature_index_sha256':hashlib.sha256((args.jobs or (args.source_features/'features.json')).read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frontend_sha256':hashlib.sha256((Path(__file__).parent/'native_mel.py').read_bytes()).hexdigest(),
        'scope':'Training/validation preparation only; no testing scores or wake accuracy acceptance',
        'jobs':len(jobs)},indent=2)+'\n')
    started=time.monotonic();counts={}
    with (args.output/'index.jsonl').open('x') as log,ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                           initargs=(args.model,args.window,args.mel_filter)) as pool:
        for number,row in enumerate(pool.map(worker,jobs,chunksize=8),1):
            log.write(json.dumps(row)+'\n');log.flush()
            key=row['split']+'/'+row['kind'];counts[key]=counts.get(key,0)+1
            if number%100==0 or number==len(jobs):
                print(json.dumps({'completed':number,'total':len(jobs),'counts':counts,
                                  'seconds':round(time.monotonic()-started,1)}),flush=True)
    (args.output/'complete.json').write_text(json.dumps({'jobs':len(jobs),'counts':counts,'seconds':time.monotonic()-started},indent=2)+'\n')


if __name__=='__main__':main()
