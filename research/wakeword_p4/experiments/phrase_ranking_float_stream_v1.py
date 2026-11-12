"""Add full float streams to the existing fixed training temporal probe."""
import argparse, collections, hashlib, json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.utils.set_random_seed(726)
assert tf.config.get_visible_devices('GPU')
sys.path.insert(0,str((base/'microWakeWord').resolve()))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
priorpath=Path('research/wakeword_p4/results/4090-phrase-ranking-temporal-probe-v1.json')
prior=json.loads(priorpath.read_text())
out={'prior_sha256':sha(priorpath),'arms':{},'limits':'Same original80 training raw records; no new data, update, threshold selection or acceptance. Exact nonstate variable multiset proves current SavedModel values match selected H5, not a historical export file hash. Float versus INT8 stream differences include runtime/numerical conversion effects, not a unique quantization mechanism. Container boundaries are synthetic references.'}
arrays={}
for arm,old in prior['arms'].items():
 root=base/f'model-phrase-ranking-{arm}-formal-v1'
 assert sha(root/'best.weights.h5')==old['weights_sha256'] and sha(root/'hey_nyx.tflite')==old['model_sha256']
 training=json.loads((root/'training.json').read_text());arch=argparse.ArgumentParser();mixednet.model_parameters(arch)
 model=mixednet.model(arch.parse_args(training['architecture_args']),(150,40),None)
 model.load_weights(root/'best.weights.h5');apply_policy(model,'frozen')
 loaded=tf.saved_model.load(str(root/'saved_model'));signature=loaded.signatures['serving_default']
 keys=list(signature.structured_input_signature[1]);assert len(keys)==1
 states=[v for v in loaded.variables if 'states' in v.name];assert states
 nonstate=[v for v in loaded.variables if 'states' not in v.name]
 def identity(v):
  a=v.numpy();return (tuple(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest())
 assert collections.Counter(map(identity,nonstate))==collections.Counter(map(identity,model.variables))
 before=[v.numpy().copy() for v in nonstate]
 def predict(raw):
  for state in states:state.assign(tf.zeros_like(state))
  values=np.clip(raw.astype(np.float32)/25.6,0,26);result=[]
  for at in range(0,len(values)-2,3):
   prediction=signature(**{keys[0]:tf.convert_to_tensor(values[None,at:at+3])})
   assert len(prediction)==1
   result.append(float(next(iter(prediction.values())).numpy().reshape(-1)[0]))
  return np.asarray(result,np.float32)
 details=[]
 for i,row in enumerate(old['details']):
  record=row['record'];path=base/'factorized-training-features-v1'/record['raw']
  assert record['split']=='training' and sha(path)==record['raw_file_sha256']
  raw=np.load(path,allow_pickle=False);assert hashlib.sha256(raw.tobytes()).hexdigest()==record['raw_sha256']
  values=predict(raw);assert np.isfinite(values).all()
  if i==0:np.testing.assert_array_equal(values,predict(raw))
  arrays[f'{arm}_{i}']=values
  selected=values[[(end-2)//3 for end in row['ends']]]
  details.append({**row,'float_stream_scores':selected.tolist(),'float_stream_mean5':float(np.mean(selected.astype(np.float64))),
    'float_stream_window_max_abs_difference':float(np.max(np.abs(selected-np.asarray(row['float_window_scores'])))),
    'float_stream_int8_max_abs_difference':float(np.max(np.abs(selected-np.asarray(row['int8_stream_scores']))))})
  if i%20==0:print(arm,i,flush=True)
 for previous,variable in zip(before,nonstate):np.testing.assert_array_equal(previous,variable.numpy())
 summary={}
 for scope in ('all','complete5'):
  rows=[r for r in details if r['record']['target']=='hey_nyx' and (scope=='all' or r['complete_container_positions']==5)]
  summary[scope]={'positive_records':len(rows),'fixed_thresholds':{}}
  for t in (.5,.992,.99609375):
   summary[scope]['fixed_thresholds'][str(t)]={mode:{'any_above':sum(max(r[mode+'_scores'])>=t for r in rows),
    'last_above':sum(r[mode+'_scores'][-1]>=t for r in rows),'mean5_above':sum(r[mode+'_mean5']>=t for r in rows)} for mode in ('float_window','float_stream','int8_stream')}
 out['arms'][arm]={'saved_model_files':{str(p.relative_to(root)):sha(p) for p in sorted((root/'saved_model').rglob('*')) if p.is_file()},
  'nonstate_variable_multiset_matches_h5':True,'nonstate_variables_unchanged':True,'first_reset_exact':True,'summary':summary,'details':details}
 print(arm,json.dumps(summary),flush=True)
arraypath=base/'phrase-ranking-float-stream-probe-v1-arrays.npz';assert not arraypath.exists()
np.savez_compressed(arraypath,**arrays)
out['arrays_sha256']=sha(arraypath);out['script_sha256']=sha(Path(__file__))
Path('research/wakeword_p4/results/4090-phrase-ranking-float-stream-v1.json').write_text(json.dumps(out,indent=2)+'\n')
