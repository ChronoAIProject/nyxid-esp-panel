"""Add float streaming to the authenticated training-only v37 crop probe."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import smooth
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/model-gpu-v37-ami-extra')
pack=Path('.local/wakeword/british-training-features-v1/training.npz')
prior=Path('.local/wakeword/v37-training-support-probe-v1')
report=json.loads((prior/'report.json').read_text())
assert sha(root/'hey_nyx.tflite')==report['model_sha256']
assert sha(root/'best.weights.h5')==report['weights_sha256']
assert sha(pack)==report['pack_sha256']
assert sha(prior/'scores.npz')==report['scores_sha256']
x=np.load(pack,allow_pickle=False)['x'];old=np.load(prior/'scores.npz',allow_pickle=False)
loaded=tf.saved_model.load(str(root/'saved_model'))
signature=loaded.signatures['serving_default'];keys=list(signature.structured_input_signature[1]);assert len(keys)==1
states=[v for v in loaded.variables if 'states' in v.name];assert states
def predict(raw):
    for state in states:state.assign(tf.zeros_like(state))
    values=np.clip(raw.astype(np.float32)/25.6,0,26);out=[]
    for at in range(0,len(values)-2,3):
        r=signature(**{keys[0]:tf.convert_to_tensor(values[None,at:at+3])})
        assert len(r)==1
        out.append(float(next(iter(r.values())).numpy().reshape(-1)[0]))
    return np.asarray(out,dtype=np.float32)
details=[];arrays={}
for i,row in enumerate(report['details']):
    f=predict(x[i]);q=old[f'stream_{i}'];assert f.shape==q.shape and np.isfinite(f).all()
    if i==0:np.testing.assert_array_equal(f,predict(x[i]))
    arrays[f'float_{i}']=f
    details.append({**row,'float_stream_last':float(f[-1]),'float_stream_max':float(max(f)),
        'float_stream_max_five_mean':float(max(smooth(f))),
        'float_score_sha256':hashlib.sha256(f.tobytes()).hexdigest(),
        'float_int8_mae':float(np.mean(np.abs(f-q))),
        'float_window_last_error':float(abs(row['window_float']-f[-1]))})
    if i%100==0:print(json.dumps({'completed':i+1,'total':len(x)}),flush=True)
out=Path('.local/wakeword/v37-float-support-probe-v1');out.mkdir(exist_ok=False)
np.savez_compressed(out/'scores.npz',**arrays)
summary={}
for label in (0,1):
    group=[d for d in details if d['label']==label];summary[str(label)]={'examples':len(group)}
    for key in ('window_float','float_stream_last','stream_last','float_stream_max','stream_max','float_stream_max_five_mean','stream_max_five_mean'):
        summary[str(label)][key]={'above_half':sum(d[key]>=.5 for d in group),
            'above_historical_threshold':sum(d[key]>=.99609375 for d in group)}
result={'scope':'Training-only crop diagnostic, no new development/threshold selection/acceptance.',
    'prior_report_sha256':sha(prior/'report.json'),'script_sha256':sha(Path(__file__)),
    'model_sha256':report['model_sha256'],'scores_sha256':sha(out/'scores.npz'),
    'saved_model_files':{str(p.relative_to(root)):sha(p) for p in sorted((root/'saved_model').rglob('*')) if p.is_file()},
    'tensorflow':tf.__version__,'first_stream_reset_exact':True,'summary':summary,
    'max_window_float_stream_last_error':max(d['float_window_last_error'] for d in details),
    'details':details,'limits':'Float and INT8 use identical isolated crops and zero model state. Last output is compared to full150-frame window without shifting phase. Five-mean includes incomplete-context startup outputs; original PCM/frontend history absent. This does not establish phase causality, full-stream accuracy, phonetic truth or generalization.'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary),flush=True)
print('max window/float-stream-last error',result['max_window_float_stream_last_error'],flush=True)
