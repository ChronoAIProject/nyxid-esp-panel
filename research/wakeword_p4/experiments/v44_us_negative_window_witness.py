"""All legal window-float endpoints for four fixed US-training stream witnesses."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
base=Path('.local/wakeword');root=base/'model-gpu-v44-ami-extra';pack=base/'us-training-features-v1';prior=base/'v44-us-training-support-v1';center=base/'v44-us-center-fit-v1';out=base/'v44-us-negative-window-witness-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'hey_nyx.tflite')=='88de16e71b5f49b5e85d226b74a07aa036324dcaab48e6f58c9c8a9343608652'
r=json.loads((prior/'report.json').read_text());c=json.loads((center/'report.json').read_text());meta=json.loads((pack/'report.json').read_text());training=json.loads((root/'training.json').read_text())
assert sha(prior/'scores.npz')==r['scores_sha256'] and sha(center/'scores.npy')==c['scores_sha256'] and sha(pack/'training.npz')==meta['pack_sha256']
selected=[d for d in r['details'] if not d['label'] and any(v['thresholds']['0.99609375']['full_high'] for v in d['modes'].values())];assert len(selected)==4
sys.path.insert(0,str((base/'microWakeWord').resolve()));from microwakeword import mixednet
p=argparse.ArgumentParser();mixednet.model_parameters(p);model=mixednet.model(p.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
out.mkdir(exist_ok=False);arrays={};details=[]
with np.load(prior/'scores.npz',allow_pickle=False) as streams:
 for d in selected:
  i=d['index'];row=meta['records'][i];assert row['pack_index']==i and row['target']!='hey_nyx' and row['split']=='training'
  raw=np.load(pack/row['raw'],allow_pickle=False);assert sha(pack/row['raw'])==row['raw_file_sha256'] and hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_sha256']
  x=np.clip(raw.astype(np.float32)/25.6,0,26);ends=np.arange(150,len(raw)+1);windows=np.stack([x[e-150:e] for e in ends]);values=np.concatenate([model(windows[a:a+64],training=False).numpy().reshape(-1) for a in range(0,len(windows),64)]);assert np.isfinite(values).all()
  arrays[f'ends_{i}']=ends;arrays[f'float_{i}']=values
  low=row['feature_crop_end'];high=min(350,len(raw));complete=((ends-150)*160<=32000)&((ends+1)*160>=32000+row['signal_samples']);assert np.array_equal(ends[complete],np.arange(low,high+1))
  modes={}
  for mode,v in d['modes'].items():
   scores=streams[f'{mode}_{i}'];assert hashlib.sha256(scores.tobytes()).hexdigest()==v['score_sha256'];five=np.convolve(scores[49:].astype(np.float64),np.ones(5)/5,'valid');assert float(five.max())==v['full_max_five']
   modes[mode]=dict(max_five=float(five.max()),peak_first_end=int(150+3*five.argmax()),peak_five_ends=(150+3*five.argmax()+3*np.arange(5)).tolist())
  stats={}
  for t in [.5,.992,.99609375]:
   mask=values>=t;stats[str(t)]=dict(high_ends=ends[mask].tolist(),high_count=int(mask.sum()),complete_high_ends=ends[mask&complete].tolist(),complete_high_count=int((mask&complete).sum()),noncomplete_high_count=int((mask&~complete).sum()))
  trained=float(values[low-150]);oldscore=c['details'][i]['score'];assert abs(trained-oldscore)<1e-5
  witness=dict(index=i,source=row['source'],voice=row['voice'],target=row['target'],variant=row['variant'],raw_sha256=row['raw_sha256'],legal_windows=len(ends),complete_ends=[int(low),int(high)],complete_count=int(complete.sum()),trained_end=low,trained_window_float=trained,prior_trained_window_float=oldscore,trained_score_batch_difference=trained-oldscore,window_float_max=float(values.max()),window_peak_end=int(ends[values.argmax()]),thresholds=stats,streams=modes)
  details.append(witness);print(json.dumps(witness),flush=True)
np.savez_compressed(out/'windows.npz',**arrays)
report=dict(scope='Four US-training negative variants whose frozen v44 actualfloat or INT8 fullstream maxfive >=.99609375; every legal150frame window scored',model_sha256=sha(root/'hey_nyx.tflite'),weights_sha256=sha(root/'best.weights.h5'),pack_report_sha256=sha(pack/'report.json'),prior_report_sha256=sha(prior/'report.json'),center_report_sha256=sha(center/'report.json'),script_sha256=sha(Path(__file__)),windows_sha256=sha(out/'windows.npz'),details=details,limits='Training-only selected witnesses, not all-negative incidence. Fullcontainer ledger is not phonetic truth or physical receptive field. Window inference/reset/phase differs from stateful stream; not an equivalence or causal proof. No gradients, threshold selection, data changes, newdev/final or promotion.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
