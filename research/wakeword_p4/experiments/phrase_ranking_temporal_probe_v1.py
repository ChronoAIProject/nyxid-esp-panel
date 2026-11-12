"""Bounded first-two-batch training-only float-window/INT8-stream diagnostic."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword'); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert tf.config.get_visible_devices('GPU')
tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
tf.keras.utils.set_random_seed(726)
sys.path.insert(0,str((base/'microWakeWord').resolve()))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
from evaluate import Detector
pack=base/'factorized-training-features-v1'
assert sha(pack/'report.json')=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
records=json.loads((pack/'report.json').read_text())['records']
probe=base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json'
assert sha(probe)=='0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f'
traces=json.loads(probe.read_text())['pair_trace'][:2]
items=[];windows=[]
for batch,tr in enumerate(traces):
 for index in tr['pack_indices']:
  record=records[index];path=pack/record['raw']
  assert record['split']=='training' and sha(path)==record['raw_file_sha256']
  raw=np.load(path,allow_pickle=False)
  assert hashlib.sha256(raw.tobytes()).hexdigest()==record['raw_sha256']
  assert raw.dtype==np.uint16 and list(raw.shape)==record['raw_shape']
  # Full-stream output cadence starts with frames0..2, hence endpoint2 mod3.
  last=tr['endpoint_inclusive']-1
  ends=list(range(last-12,last+1,3));assert all(e%3==2 and e>=149 for e in ends)
  slices=np.stack([raw[e-149:e+1] for e in ends]);assert slices.shape==(5,150,40)
  windows.extend(slices)
  items.append({'batch':batch,'record':record,'ends':ends,'raw':raw,
    'complete_container_positions':sum((e-150)*160<=32000 and (e+1)*160>=32000+record['signal_samples'] for e in ends)})
x=np.clip(np.stack(windows).astype(np.float32)/25.6,0,26)
out={'scope':'First two original paired training batches, all80 records/five renderings, no selection by scores',
 'probe_sha256':sha(probe),'pack_report_sha256':sha(pack/'report.json'),'arms':{},
 'limits':'Reused training-only synthetic container boundaries, not phonetic truth. Float isolated windows vs full-context INT8 streams: differences include quantization/context effects; not uniquely quantization. Alignment intentionally uses actual stream endpoints2 mod3, one frame before original paired endpoint0 mod3. Five scores can include incomplete-container windows; report support count. No threshold tuning, optimization, new dev/final or acceptance.'}
for arm in ('control','treatment'):
 root=base/f'model-phrase-ranking-{arm}-formal-v1';training=json.loads((root/'training.json').read_text())
 assert sha(root/'hey_nyx.tflite')==training['model_sha256']
 audit=json.loads(Path(f'research/wakeword_p4/results/4090-phrase-ranking-{arm}-formal-v1-export-audit.json').read_text())
 assert sha(root/'best.weights.h5') in json.dumps(audit)
 arch=argparse.ArgumentParser();mixednet.model_parameters(arch)
 model=mixednet.model(arch.parse_args(training['architecture_args']),(150,40),None)
 model.load_weights(root/'best.weights.h5');apply_policy(model,'frozen')
 before=[v.numpy().copy() for v in model.variables]
 predictions=np.concatenate([model(x[start:start+40],training=False).numpy().reshape(-1) for start in range(0,len(x),40)]).reshape(-1,5)
 detector=Detector(root/'hey_nyx.tflite');details=[]
 for i,item in enumerate(items):
  detector.reset();stream=detector.predict(item['raw'])
  values=stream[[(e-2)//3 for e in item['ends']]]
  assert len(values)==5 and np.isfinite(values).all()
  details.append({k:v for k,v in item.items() if k!='raw'} | {'float_window_scores':predictions[i].tolist(),'int8_stream_scores':values.tolist(),
   'float_window_mean5':float(np.mean(predictions[i].astype(np.float64))),'int8_stream_mean5':float(np.mean(values.astype(np.float64)))})
 for previous,variable in zip(before,model.variables):np.testing.assert_array_equal(previous,variable.numpy())
 summary={}
 for label in ('hey_nyx','negative'):
  rows=[r for r in details if (r['record']['target']=='hey_nyx')==(label=='hey_nyx')]
  summary[label]={'records':len(rows),'complete_container_positions':sum(r['complete_container_positions'] for r in rows)}
  for mode in ('float_window','int8_stream'):
   summary[label][mode]={str(t):{'last_above':sum(r[mode+'_scores'][-1]>=t for r in rows),
    'any_above':sum(max(r[mode+'_scores'])>=t for r in rows),'mean5_above':sum(r[mode+'_mean5']>=t for r in rows)} for t in (.5,.992,.99609375)}
 out['arms'][arm]={'model_sha256':training['model_sha256'],'weights_sha256':sha(root/'best.weights.h5'),'all_variables_unchanged':True,'summary':summary,'details':details}
 print(arm,json.dumps(summary),flush=True)
out['script_sha256']=sha(Path(__file__))
Path('research/wakeword_p4/results/4090-phrase-ranking-temporal-probe-v1.json').write_text(json.dumps(out,indent=2)+'\n')
