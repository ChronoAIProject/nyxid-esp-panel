"""All eligible old-source training variants: window fit vs full INT8 history."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
from evaluate import Detector,smooth
root=Path('.local/wakeword/model-gpu-v40-ami-extra')
pack=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'hey_nyx.tflite')=='0b48cd96e8837b27c2df73c709c3360da9f06079f61eb9464922bae438bb98e4'
training=json.loads((root/'training.json').read_text());coverage=CompleteCoverage(pack)
indices=sorted({3*i+v for identities in coverage.groups.values() for ids in identities.values() for i in ids for v in range(3)})
assert len(indices)==6597
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
model=mixednet.model(parser.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
float_scores=np.concatenate([model(np.clip(coverage.x[indices[a:a+64]].astype(np.float32)/25.6,0,26),training=False).numpy().reshape(-1) for a in range(0,len(indices),64)])
detector=Detector(root/'hey_nyx.tflite');out=Path('.local/wakeword/v40-training-support-v1');out.mkdir(exist_ok=False)
arrays={'window_float':float_scores,'pack_indices':np.array(indices)};details=[]
for j,i in enumerate(indices):
 r=coverage.records[i];raw=np.load(pack/r['raw'],allow_pickle=False);assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
 detector.reset();q=detector.predict(raw);assert np.isfinite(q).all()
 if j==0:
  detector.reset();np.testing.assert_array_equal(q,detector.predict(raw))
 arrays[f'int8_{i}']=q
 # Only scores with a complete original150-frame context; retain full history.
 complete=q[49:];five=np.convolve(complete.astype(np.float64),np.ones(5)/5,mode='valid')
 details.append(dict(pack_index=i,engine=r['source_group'],phrase=coverage.sources[r['source_index']]['aliases'][0]['phrase'],label=r['label'],variant=r['variant'],window_float=float(float_scores[j]),full_int8_max=float(complete.max()),full_int8_max_five=float(five.max()),score_sha256=hashlib.sha256(q.tobytes()).hexdigest()))
 if j%500==0:print(json.dumps({'completed':j+1,'total':len(indices)}),flush=True)
np.savez_compressed(out/'scores.npz',**arrays)
summary={}
for d in details:
 key=d['engine']+'/'+('positive' if d['label'] else d['phrase'])
 g=summary.setdefault(key,dict(examples=0,window_half=0,window_historical=0,full_five_half=0,full_five_historical=0));g['examples']+=1
 for name,score,t in [('window_half',d['window_float'],.5),('window_historical',d['window_float'],.99609375),('full_five_half',d['full_int8_max_five'],.5),('full_five_historical',d['full_int8_max_five'],.99609375)]:g[name]+=int(score>=t)
report=dict(scope='All6597 old-source variants eligible for v40 sampling; reused training-only diagnostic',model_sha256=sha(root/'hey_nyx.tflite'),training_sha256=sha(root/'training.json'),pack_sha256=sha(pack/'training.npz'),script_sha256=sha(Path(__file__)),scores_sha256=sha(out/'scores.npz'),first_reset_exact=True,summary=summary,details=details,limits='Window float vs full INT8 confounds quantization/phase/state; not isolated causal evidence or phonetic recall. Full max-five is descriptive, not latched-event accuracy or a new threshold. Excludes395 non-task canonical rows as preregistered sampler; no new/final/model training.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)
