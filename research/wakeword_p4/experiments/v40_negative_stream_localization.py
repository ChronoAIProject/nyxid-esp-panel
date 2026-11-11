"""Same checkpoint full-history float vs INT8 for all eligible training sources."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from sequence_scorer import build_sequence_scorer
root=Path('.local/wakeword/model-gpu-v40-ami-extra');pack=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study');prior=Path('.local/wakeword/v40-training-support-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((prior/'report.json').read_text());assert sha(root/'hey_nyx.tflite')==report['model_sha256'] and sha(prior/'scores.npz')==report['scores_sha256']
training=json.loads((root/'training.json').read_text());records=json.loads((pack/'records.json').read_text());sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
window=mixednet.model(parser.parse_args(training['architecture_args']),(150,40),None);window.load_weights(root/'best.weights.h5')
for layer in window.layers:
 if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
sequence=build_sequence_scorer(window);before=[v.numpy().copy() for v in window.variables]
loaded=tf.saved_model.load(str(root/'saved_model'));signature=loaded.signatures['serving_default'];key=next(iter(signature.structured_input_signature[1]));states=[v for v in loaded.variables if 'states' in v.name];assert states

previous=Path('.local/wakeword/v40-float-history-v1')
float_report=json.loads((previous/'report.json').read_text())
assert sha(previous/'scores.npz')==float_report['scores_sha256']
floats=np.load(previous/'scores.npz'); arrays={}; details=[]
# Diagnostic census of the union, not a new acceptance set or filtered training set.
selected=[d for d in float_report['details'] if not d['label'] and
          (d['float_full_max_five']>=.5 or d['full_int8_max_five']>=.5)]
for j,d in enumerate(selected):
 i=d['pack_index'];r=records[i];raw=np.load(pack/r['raw']);n=len(raw)//3*3
 assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
 x=np.clip(raw[:n].astype(np.float32)/25.6,0,26)[None]
 for state in states:state.assign(tf.zeros_like(state))
 reference=[]
 for at in range(0,n,3):
  reference.append(float(next(iter(signature(**{key:tf.convert_to_tensor(x[:,at:at+3])}).values())).numpy().reshape(-1)[0]))
 actual=np.array(reference,dtype=np.float32)[49:];f=floats[f'float_{i}']
 assert actual.shape==f.shape
 five=np.convolve(actual.astype(np.float64),np.ones(5)/5,mode='valid');peak=int(five.argmax())
 # Output49 ends at raw frame150 (exclusive); each later output advances3.
 end=(49+peak+4+1)*3
 arrays[f'stream_float_{i}']=actual
 details.append({**d,'actual_stream_max_five':float(five[peak]),
  'sequence_stream_max_error':float(np.abs(actual-f).max()),
  'peak_five_end_feature_exclusive':end,'training_crop_end':r['feature_crop_end'],
  'peak_minus_crop_end_frames':end-r['feature_crop_end']})
 print(json.dumps({'completed':j+1,'total':len(selected)}),flush=True)
for v,b in zip(window.variables,before):np.testing.assert_array_equal(v.numpy(),b)
out=Path('.local/wakeword/v40-negative-stream-localization-v1');out.mkdir(exist_ok=False)
np.savez_compressed(out/'scores.npz',**arrays)
result=dict(scope='Diagnostic union of training negatives >=.5 in prior float sequence or INT8 complete history',
 selection_count=len(selected),model_sha256=sha(root/'hey_nyx.tflite'),
 prior_report_sha256=sha(previous/'report.json'),script_sha256=sha(Path(__file__)),
 scores_sha256=sha(out/'scores.npz'),optimizer_updates=0,all_window_variables_unchanged=True,
 summary={str(t):sum(d['actual_stream_max_five']>=t for d in details) for t in [.5,.99609375]},
 max_sequence_stream_error=max(d['sequence_stream_max_error'] for d in details),details=details,
 limits='Selected training diagnostic only; no new acceptance, causal proof, phonetic boundary or independent evaluation. Feature end mapping is processing support, not acoustic alignment.')
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']),flush=True)
