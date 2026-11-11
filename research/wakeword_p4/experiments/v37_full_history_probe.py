"""Reconstruct training streams and audit known phase1 alignment, no new data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
from evaluate import smooth
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/model-gpu-v37-ami-extra');pack=Path('.local/wakeword/british-training-features-v1')
assert sha(pack/'training.npz')=='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
assert sha(root/'hey_nyx.tflite')=='09c5191e9b8a1ebe5ebfdfd62e22ea49f520d0eae4fe826910ec614187705d06'
training=json.loads((root/'training.json').read_text());report=json.loads((pack/'report.json').read_text())
data=np.load(pack/'training.npz',allow_pickle=False)['x']
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
p=argparse.ArgumentParser();mixednet.model_parameters(p)
model=mixednet.model(p.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
loaded=tf.saved_model.load(str(root/'saved_model'));signature=loaded.signatures['serving_default']
keys=list(signature.structured_input_signature[1]);assert len(keys)==1
states=[v for v in loaded.variables if 'states' in v.name];assert states
roots={'kokoro':Path('.local/wakeword/synthetic-phone-training-v1'),'alba':Path('.local/wakeword/alba-training-pilot-v2'),'cori':Path('.local/wakeword/cori-training-pilot-v1')}
out=Path('.local/wakeword/v37-full-history-probe-v1');out.mkdir(exist_ok=False)
ir=np.zeros(961);ir[[0,160,480,960]]=[1,.25,-.12,.06]
details=[];arrays={};threshold=.99609375
for row in report['records']:
    i=row['pack_index']
    if i is None:continue
    path=roots[row['source_group']]/row['source'];assert sha(path)==row['source_sha256']
    pcm,rate=sf.read(path,dtype='float64');assert rate==16000 and pcm.ndim==1
    speech=fftconvolve(pcm,ir,mode='full') if row['variant']==2 else pcm.copy()
    signal=speech*(.08/max(1e-8,float(np.sqrt(np.mean(speech**2)))))
    wave=np.pad(signal,(32000,16000));rng=np.random.default_rng(row['seed'])
    if row['variant']:
        noise=rng.normal(0,1,len(wave));noise*=.08*10**(-row['snr_db']/20)/np.sqrt(np.mean(noise**2));wave+=noise
    wave*=min(1.,.95/max(float(np.max(np.abs(wave))),1e-8))
    raw=features(wave*32768);e=row['feature_crop_end']
    np.testing.assert_array_equal(raw[e-150:e],data[i])
    x=np.clip(raw.astype(np.float32)/25.6,0,26)
    for state in states:state.assign(tf.zeros_like(state))
    f=[]
    for at in range(0,len(x)-2,3):
        r=signature(**{keys[0]:tf.convert_to_tensor(x[None,at:at+3])})
        f.append(float(next(iter(r.values())).numpy().reshape(-1)[0]))
    f=np.asarray(f,dtype=np.float32)
    ends=np.asarray([v for v in range(150,len(x),3)])
    windows=np.stack([x[v-149:v+1] for v in ends])
    aligned=np.concatenate([model(windows[j:j+64],training=False).numpy().reshape(-1) for j in range(0,len(windows),64)])
    ref=f[ends//3-1];error=np.abs(aligned-ref)
    arrays[f'raw_{i}']=raw;arrays[f'float_{i}']=f;arrays[f'aligned_{i}']=aligned
    details.append({'index':i,'target':row['target'],'source_group':row['source_group'],
        'source':row['source'],'variant':row['variant'],'crop_exact':True,
        'aligned_window_stream_max_error':float(max(error)),
        'aligned_window_stream_mae':float(np.mean(error)),
        'float_max':float(max(f)),'float_max_five_mean':float(max(smooth(f))),
        'crop_end_mod3':e%3,'raw_sha256':hashlib.sha256(raw.tobytes()).hexdigest()})
    if i%100==0:print(json.dumps({'completed':i+1,'total':len(data)}),flush=True)
np.savez_compressed(out/'arrays.npz',**arrays)
summary={}
for target in sorted({d['target'] for d in details}):
    group=[d for d in details if d['target']==target]
    summary[target]={'examples':len(group),'max_above_threshold':sum(d['float_max']>=threshold for d in group),
        'five_mean_above_threshold':sum(d['float_max_five_mean']>=threshold for d in group)}
r={'scope':'All included training sources only; reconstructed exact crops and original history, no new development/final data.',
    'model_sha256':sha(root/'hey_nyx.tflite'),'pack_sha256':sha(pack/'training.npz'),
    'script_sha256':sha(Path(__file__)),'arrays_sha256':sha(out/'arrays.npz'),
    'phase':1,'summary':summary,'all_crops_exact':len(details)==1002 and all(d['crop_exact'] for d in details),
    'max_aligned_window_stream_error':max(d['aligned_window_stream_max_error'] for d in details),
    'details':details,'limits':'Training reuse only. Full generated signal includes native silence/RIR, not phonetic truth. Fixed historical threshold, no selection or acceptance. Max/five-mean counts are descriptive over whole stream; no event policy or generalization claim.'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='details'}),flush=True)
