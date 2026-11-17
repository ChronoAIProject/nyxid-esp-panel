"""Fixed original batch128 float/INT8 fit diagnostic, no optimization."""
import gzip,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-training-fit-v1')
T=Path('.local/wakeword/binary40-training-v1');LOCK=R/'4090-binary40-training-fit-plan-v1.json'
def plan():
    files=[Path(__file__),T/'final.pt',T/'batches.jsonl.gz',R/'4090-binary40-training-original-v1.json',R/'4090-binary40-trained-export-original-v1.json',
           Path('.local/wakeword/binary40-trained-int8-v1/int8.tflite'),Path('research/wakeword_p4/training/causal_binary40_v1.py'),
           Path(__file__).with_name('binary40_stream_predict_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py')]
    original=json.loads((R/'4090-binary40-training-plan-v1.json').read_text())
    hashes=dict(original['files']);hashes.update({str(f):sha(f) for f in files})
    p=dict(files=hashes,seed=726,rows=128,threshold=.91796875,
           policy='Final20000 only, original Sampler726 first full128 batch, original feature/mask/descriptor hashes verified against savedstep1. GPUfloat and BUILTIN_REF1thread INT8 complete282calls perrow, per-layer realzero cache startup exactly as training. Float five-probability event policy versus original exactUINT8 policy; keep every crossing/event and positive mask-contained fivecall crossing. All128rows, no selection/retry/optimization/threshold/calibration rescue or dev/acceptance input. Cached training features are not whole primed PCM. First16rows overlap fixed export calibration; report that dependence, not independent quantization validation.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def float_events(v):
    events=[];last=-1000;armed=True
    for j in range(4,len(v)):
        mean=float(np.asarray(v[j-4:j+1],np.float64).sum()/5)
        step=j+1
        if mean<.91796875*.5 and step-last>=67:armed=True
        if armed and mean>=.91796875 and step-last>=67:events.append(step);last=step;armed=False
    return events
def crossing(v,mask,integer):
    sums=np.convolve(np.asarray(v,np.int64 if integer else np.float64),np.ones(5,dtype=np.int64),'valid')
    contained=np.convolve(mask.astype(np.int64),np.ones(5,dtype=np.int64),'valid')==5
    ends=np.flatnonzero(contained & (sums >= (1175 if integer else 1175/256)))+5
    return ends.tolist()
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for f,h in p['files'].items():assert sha(f)==h,f
    B.mkdir(exist_ok=False)
    import torch
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    from causal_source_balanced_batch_v1 import Sampler,digest
    from binary40_stream_predict_v1 import predict
    from uint8_event_policy_v1 import event_steps
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_v1 import Binary40
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    cp=torch.load(T/'final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
    model=Binary40().cuda().eval();model.load_state_dict(cp['model'])
    (x,pw,nw),descriptors=Sampler(726).batch()
    with gzip.open(T/'batches.jsonl.gz','rt') as f:original=json.loads(next(f))
    assert descriptors==original['descriptors']
    for key,values in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(values)==original[key]
    start=time.monotonic()
    with torch.no_grad():
        logits,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(128,'cuda'))
        floats=torch.sigmoid(logits[:,0,:]).cpu().numpy()
    def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
    quantized=np.empty_like(floats,dtype=np.uint8);records=[];groups={}
    export=json.loads((R/'4090-binary40-trained-export-original-v1.json').read_text())
    for i,d in enumerate(descriptors):
        _,codes,q=predict(x[i],export['model']['path'],runtime);quantized[i]=codes
        fe=float_events(floats[i]);qe=event_steps(codes,.91796875)
        fc=crossing(floats[i],pw[i]>0,False);qc=crossing(codes,pw[i]>0,True)
        positive=bool(pw[i].sum()>0)
        def inside(es,mask):return [s for s in es if mask[s-1]>0]
        record=dict(index=i,source=d,positive=positive,calibration_overlap=i<16,
                    float_events=fe,int8_events=qe,float_positive_mask_crossings=fc,int8_positive_mask_crossings=qc,
                    float_positive_events=inside(fe,pw[i]),int8_positive_events=inside(qe,pw[i]),
                    float_negative_events=inside(fe,nw[i]),int8_negative_events=inside(qe,nw[i]),
                    max_probability_difference=float(np.abs(floats[i].astype(np.float64)-codes.astype(np.float64)/256).max()),quantization=q)
        records.append(record)
        key=d['kind']+'/'+('positive' if positive else 'negative');g=groups.setdefault(key,dict(rows=0,float_positive_crossings=0,int8_positive_crossings=0,float_event_rows=0,int8_event_rows=0,float_negative_event_rows=0,int8_negative_event_rows=0));g['rows']+=1
        g['float_positive_crossings']+=bool(fc);g['int8_positive_crossings']+=bool(qc);g['float_event_rows']+=bool(fe);g['int8_event_rows']+=bool(qe)
        g['float_negative_event_rows']+=bool(record['float_negative_events']);g['int8_negative_event_rows']+=bool(record['int8_negative_events'])
    trace=B/'all128.npz';np.savez_compressed(trace,float_probability=floats,uint8=quantized,positive_weights=pw,negative_weights=nw)
    out=dict(schema='binary40-training-fit-original-v1',plan_sha256=sha(LOCK),checkpoint_sha256=sha(T/'final.pt'),model_sha256=export['model']['sha256'],
             rows=128,positive_rows=sum(r['positive'] for r in records),calibration_overlapping_rows=16,groups=groups,records=records,
             trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),
             optimizer_updates=0,policy=p['policy'])
    with (R/'4090-binary40-training-fit-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(groups=groups,rows=128,positive_rows=out['positive_rows'],elapsed_seconds=out['elapsed_seconds'])))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
