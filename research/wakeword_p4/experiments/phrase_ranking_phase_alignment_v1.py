"""Apply the existing phase=1 contract to the frozen training temporal probe."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(726)
assert tf.config.get_visible_devices('GPU')
sys.path.insert(0,str((base/'microWakeWord').resolve()))
from microwakeword import mixednet
source=Path('research/wakeword_p4/results/4090-phrase-ranking-float-stream-v1.json');prior=json.loads(source.read_text())
out={'source_sha256':sha(source),'arms':{},'limits':'Correction/extension of unaligned window baseline; original results preserved. Existing training metadata phase=1, not fitted to these scores. Float phase agreement does not prove generalization, eliminate INT8 degradation, or accept either candidate. Same reused80 training records only.'}
for arm,old in prior['arms'].items():
 root=base/f'model-phrase-ranking-{arm}-formal-v1';training=json.loads((root/'training.json').read_text());phase=training['validation_window_phase'];assert phase==1
 audit=json.loads(Path(f'research/wakeword_p4/results/4090-phrase-ranking-{arm}-formal-v1-export-audit.json').read_text());assert sha(root/'best.weights.h5')==audit['weights_sha256']
 arch=argparse.ArgumentParser();mixednet.model_parameters(arch)
 model=mixednet.model(arch.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
 windows=[]
 for row in old['details']:
  record=row['record'];path=base/'factorized-training-features-v1'/record['raw'];assert sha(path)==record['raw_file_sha256']
  raw=np.load(path,allow_pickle=False);assert hashlib.sha256(raw.tobytes()).hexdigest()==record['raw_sha256']
  windows.extend(raw[end-149+phase:end+1+phase] for end in row['ends'])
 x=np.clip(np.stack(windows).astype(np.float32)/25.6,0,26);assert x.shape==(400,150,40)
 predictions=np.concatenate([model(x[i:i+40],training=False).numpy().reshape(-1) for i in range(0,len(x),40)]).reshape(80,5)
 reference=np.asarray([row['float_stream_scores'] for row in old['details']])
 delta=np.abs(predictions-reference)
 passed=bool(np.allclose(predictions,reference,rtol=1e-5,atol=1e-5))
 out['arms'][arm]={'phase':phase,'weights_sha256':sha(root/'best.weights.h5'),'aligned_window_scores':predictions.tolist(),'max_abs_difference':float(delta.max()),'mean_abs_difference':float(delta.mean()),'allclose_rtol1e-5_atol1e-5':passed}
 print(arm,out['arms'][arm]['max_abs_difference'],passed,flush=True)
out['script_sha256']=sha(Path(__file__))
Path('research/wakeword_p4/results/4090-phrase-ranking-phase-alignment-v1.json').write_text(json.dumps(out,indent=2)+'\n')
