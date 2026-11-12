"""Training-only window versus actual float streaming on all12 INT8-high witnesses."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');root=base/'model-gpu-v42-ami-extra';pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'hey_nyx.tflite')=='ff15ebf95001433135514babe0929bb75afa79d0e2a9755e0d61d9a6c7d81272'
r=json.loads((base/'v42-training-support-v1/report.json').read_text());records=json.loads((pack/'records.json').read_text());training=json.loads((root/'training.json').read_text())
sys.path.insert(0,str((base/'microWakeWord').resolve()));from microwakeword import mixednet
parser=argparse.ArgumentParser();mixednet.model_parameters(parser);model=mixednet.model(parser.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
loaded=tf.saved_model.load(str(root/'saved_model'));sig=loaded.signatures['serving_default'];key=next(iter(sig.structured_input_signature[1]));states=[v for v in loaded.variables if 'states' in v.name];assert states
witnesses=[]
for d in r['details']:
 if d['label'] or d['full_int8_max_five']<.5:continue
 i=d['pack_index'];record=records[i];raw=np.load(pack/record['raw']);assert hashlib.sha256(raw.tobytes()).hexdigest()==record['raw_sha256'];x=np.clip(raw.astype(np.float32)/25.6,0,26)
 ends=np.arange(150,len(raw)+1);windows=np.stack([x[e-150:e] for e in ends]);scores=np.concatenate([model(windows[a:a+64],training=False).numpy().reshape(-1) for a in range(0,len(windows),64)])
 for state in states:state.assign(tf.zeros_like(state))
 stream=[]
 for a in range(0,len(x)-2,3):stream.append(float(next(iter(sig(**{key:tf.convert_to_tensor(x[None,a:a+3])}).values())).numpy().reshape(-1)[0]))
 mature=np.asarray(stream,dtype=np.float32)[49:];five=np.convolve(mature.astype(np.float64),np.ones(5)/5,'valid')
 row=dict(pack_index=i,engine=d['engine'],phrase=d['phrase'],variant=d['variant'],legal_uniform_ends=len(ends),center_window_float=d['window_float'],int8_max_five=d['full_int8_max_five'],actual_float_max_five=float(five.max()),window_float_max=float(scores.max()),window_half_count=int((scores>=.5).sum()),window_historical_count=int((scores>=.992).sum()),actual_float_peak_first_end=int(150+3*five.argmax()),window_float_peak_end=int(ends[scores.argmax()]))
 witnesses.append(row);print(json.dumps(row),flush=True)
assert len(witnesses)==12
out=dict(scope='All12 v42 eligible old-training negatives with saved full INT8 max-five >=.5; selected training diagnostic only',model_sha256=sha(root/'hey_nyx.tflite'),prior_report_sha256=sha(base/'v42-training-support-v1/report.json'),script_sha256=sha(Path(__file__)),witnesses=witnesses,limits='All legal negative windows evaluated without optimization or filtering. Window float versus actual streaming contexts/phases differ; not exact-equivalence evidence. No new/final data, threshold selection or acceptance. Window counts describe uniform draw probability, not realized training exposure.')
Path('research/wakeword_p4/results/4090-v42-negative-window-witness-v1.json').write_text(json.dumps(out,indent=2)+'\n')
