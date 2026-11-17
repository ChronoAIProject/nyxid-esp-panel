"""Fixed first builtin physical stream: native3/CUDAchunk/CPU Keras parity."""
import json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');LOCK=R/'4090-binary40-physical-float-check-plan-v2.json'
def plan():
    replay=Path('.local/wakeword/binary40-float-replay-v2/builtin/result.json');d=json.loads(replay.read_text())
    files=[Path(__file__),replay,Path(d['records'][0]['trace']['path']),R/'4090-binary40-builtin-replay-plan-v1.json',
           Path('.local/wakeword/binary40-training-v1/final.pt'),Path('research/wakeword_p4/training/causal_binary40_v1.py'),Path(__file__).with_name('export_binary40_feasibility_v1.py')]
    p=dict(files={str(f):sha(f) for f in files},policy='First original builtin clip index0 fixed, whole81600zeroPCM+audio stream, all native3calls CUDA and CPUKeras compared to saved3000frameCUDAchunk probabilities; all caches paired CUDA/Keras. No optimization, new calibration, threshold or recognition retuning. Implementation gate only, not additional exposure.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    import torch,tensorflow as tf,soundfile as sf
    from export_binary40_feasibility_v1 import build
    from collect_multispeaker_formal_v1 import prepare
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_v1 import Binary40
    from features import features
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tf.config.set_visible_devices([],'GPU');assert torch.cuda.is_available()
    cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
    model=Binary40().cuda().eval();model.load_state_dict(cp['model']);keras,histories,channels=build(cp['model'])
    original=json.loads((R/'4090-binary40-builtin-replay-plan-v1.json').read_text())['rows'][0]
    wav,=[f for f in original['files'] if f['path'].endswith('.wav')];assert sha(wav['path'])==wav['sha256']
    pcm,rate=sf.read(wav['path'],dtype='int16');assert rate==16000
    combined=np.concatenate((np.zeros(81600,np.int16),pcm));values,support,_=prepare(features(combined),len(combined))
    replay=json.loads(Path('.local/wakeword/binary40-float-replay-v2/builtin/result.json').read_text())['records'][0]
    with np.load(replay['trace']['path']) as z:saved=z['float_probability'].copy()
    states=model.initial_state(device='cuda');ks={f'cache{i}':np.zeros((1,h,1,c),np.float32) for i,(h,c) in enumerate(zip(histories,channels))}
    outputs=[];keras_error=0.;cache_error=0.
    with torch.no_grad():
        for start in range(0,len(values),3):
            chunk=values[start:start+3].T.copy()[None];y,states=model(torch.tensor(chunk,device='cuda'),states);prob=float(torch.sigmoid(y)[0,0,0]);outputs.append(prob)
            result=keras(dict(chunk=chunk.transpose(0,2,1)[:,:,None,:],**ks),training=False)
            actual=float(result['probability'].numpy()[0,0,0,0]);keras_error=max(keras_error,abs(actual-prob));assert np.isclose(actual,prob,atol=2e-4,rtol=2e-4)
            for i,v in enumerate(states):
                ks[f'cache{i}']=result[f'cache{i}_out'].numpy();expected=v.cpu().numpy().transpose(0,2,1)[:,:,None,:]
                cache_error=max(cache_error,float(np.abs(ks[f'cache{i}']-expected).max()));assert np.allclose(ks[f'cache{i}'],expected,atol=2e-4,rtol=2e-4)
    assert np.allclose(outputs,saved,atol=2e-4,rtol=2e-4)
    out=dict(schema='binary40-physical-float-check-original-v2',plan_sha256=sha(LOCK),calls=len(outputs),native3_vs_saved_chunk_max_probability_error=float(np.max(np.abs(np.array(outputs)-saved))),cuda_vs_keras_max_probability_error=keras_error,max_cache_error=cache_error,all_checks_passed=True,policy=p['policy'])
    with (R/'4090-binary40-physical-float-check-original-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
