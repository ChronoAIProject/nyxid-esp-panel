"""Full training-wave smoke INT8 reset/stream contract; no recognition gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter, OpResolverType

RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-broad-int8-probe-plan-v1.json'
OUT=Path('.local/wakeword/hey-broad-int8-probe-v1')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plan():
    original=RESULTS/'4090-hey-context225-stream-probe-plan-v3.json'
    requests=json.loads(original.read_text())['requests']
    models={t:str(Path('.local/wakeword/hey-broad-smoke-v1')/t/'hey_nyx.tflite') for t in ('hey_willow','hey_nyx')}
    files={str(p):sha(p) for p in [Path(__file__),original,Path('research/wakeword_p4/training/features.py')]}
    files.update({p:sha(p) for p in models.values()})
    for r in requests: files[r['path']]=r['sha256']
    d=dict(schema='hey-broad-int8-probe-plan-v1',files=files,requests=requests,models=models,
        policy='All six previously bound full training PCM through each smoke model. Lead48000/tail16000. One full frontend traversal per input, final-only160sample padding then final-only3feature zero padding. Fresh interpreter twice; exact full ordered uint8 traces. No thresholds, selection, recognition/FAR or heldout input.',
        limits='Structural reset/stream smoke only; not float-INT8 accuracy or formal trained candidate admission.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')

def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    for p,h in d['files'].items(): assert sha(p)==h
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    assert not OUT.exists();OUT.mkdir()
    cases=[]
    for i,r in enumerate(d['requests']):
        pcm,rate=sf.read(r['path'],dtype='int16');assert rate==16000 and pcm.ndim==1
        pcm=np.pad(pcm,(48000,16000));raw=features(pcm)
        raw_pad=np.pad(raw,((0,(-len(raw))%3),(0,0)))
        values=np.clip(raw_pad.astype(np.float32)/25.6,0,26)
        for target,path in d['models'].items():
            traces=[]
            for repeat in range(2):
                interpreter=Interpreter(model_path=path,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1)
                interpreter.allocate_tensors();inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
                assert list(inp['shape'])==[1,3,40] and inp['dtype']==np.int8
                assert list(out['shape'])==[1,1] and out['dtype']==np.uint8
                scale,zero=inp['quantization'];oscale,ozero=out['quantization'];assert scale>0 and oscale>0
                q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
                trace=[]
                for at in range(0,len(q),3):
                    interpreter.set_tensor(inp['index'],q[None,at:at+3]);interpreter.invoke()
                    trace.append(int(interpreter.get_tensor(out['index'])[0,0]))
                traces.append(np.asarray(trace,dtype=np.uint8))
            assert np.array_equal(*traces) and len(traces[0])==len(raw_pad)//3
            clocks=np.minimum((3*np.arange(1,len(traces[0])+1)+2)*160,len(pcm)).astype('<i8')
            p=OUT/f'{i:02}-{target}.npz'
            np.savez(p,raw=raw,quantized_input=q,output=traces[0],sample_end=clocks)
            cases.append(dict(source=r,target=target,pcm_samples=len(pcm),feature_frames=len(raw),final_feature_padding=len(raw_pad)-len(raw),calls=len(traces[0]),twice_fresh_reset_exact=True,arrays_path=str(p),arrays_sha256=sha(p),input_quantization=[scale,zero],output_quantization=[oscale,ozero]))
    result=dict(schema='hey-broad-int8-probe-v1',plan_sha256=sha(PLAN),cases=cases,limits=d['limits'])
    with (RESULTS/'4090-hey-broad-int8-probe-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),all_fresh_resets_exact=True,total_calls=sum(c['calls'] for c in cases))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
