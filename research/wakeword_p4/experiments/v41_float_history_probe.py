"""Same checkpoint full-history float vs INT8 for all eligible training sources."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from sequence_scorer import build_sequence_scorer
root=Path('.local/wakeword/model-gpu-v41-ami-extra');pack=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study');prior=Path('.local/wakeword/v41-training-support-v1')
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
original=np.load(prior/'scores.npz');arrays={};details=[];seen=set();witness=[]
for j,d in enumerate(report['details']):
 i=d['pack_index'];r=records[i];raw=np.load(pack/r['raw']);assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256'];n=len(raw)//3*3
 x=np.clip(raw[:n].astype(np.float32)/25.6,0,26)[None];f=sequence(x[:,1:],training=False).numpy()[0];q=original[f'int8_{i}'][49:];assert f.shape==q.shape
 group=(d['engine'],'positive' if d['label'] else d['phrase'])
 if group not in seen:
  seen.add(group)
  for state in states:state.assign(tf.zeros_like(state))
  reference=[]
  for at in range(0,n,3):reference.append(float(next(iter(signature(**{key:tf.convert_to_tensor(x[:,at:at+3])}).values())).numpy().reshape(-1)[0]))
  witness.append(dict(pack_index=i,outputs=len(f),max_sequence_stream_error=float(np.abs(f-np.array(reference[49:])).max())))
 five=np.convolve(f.astype(np.float64),np.ones(5)/5,mode='valid');arrays[f'float_{i}']=f
 details.append({**d,'float_full_max_five':float(five.max()),'float_int8_mae':float(np.abs(f-q).mean()),'float_int8_max_error':float(np.abs(f-q).max()),'float_score_sha256':hashlib.sha256(f.tobytes()).hexdigest()})
 if j%500==0:print(json.dumps({'completed':j+1,'total':len(report['details'])}),flush=True)
for v,b in zip(window.variables,before):np.testing.assert_array_equal(v.numpy(),b)
out=Path('.local/wakeword/v41-float-history-v1');out.mkdir(exist_ok=False);np.savez_compressed(out/'scores.npz',**arrays)
summary={}
for label in [0,1]:
 rows=[d for d in details if d['label']==label];g={'examples':len(rows)}
 for t in [.5,.99609375]:
  g[str(t)]={k:sum(d[k]>=t for d in rows) for k in ['window_float','float_full_max_five','full_int8_max_five']}
 summary[str(label)]=g
result=dict(scope='All6597 eligible training variants; same v41 checkpoint and saved histories',model_sha256=report['model_sha256'],prior_report_sha256=sha(prior/'report.json'),weights_sha256=sha(root/'best.weights.h5'),script_sha256=sha(Path(__file__)),scores_sha256=sha(out/'scores.npz'),all_window_variables_unchanged=True,optimizer_updates=0,sequence_stream_witnesses=witness,summary=summary,details=details,limits='Phase1 shared float sequence validated against exported float streaming on first engine/task witnesses only; all residuals retained. Not phonetic/event accuracy, new acceptance or independent final. No training/threshold selection/new data.')
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
