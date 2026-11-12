"""All-arm frontend input-range census; no wake model or data selection."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');root=base/'balanced-rendering-factorial-v1';pack=base/'piper-balanced-training-features-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads(Path('research/wakeword_p4/results/4090-balanced-rendering-controls-audit-v1.json').read_text());assert audit['all_arrays_pcm_frontend_exact'] and sha(root/'controls.json')==audit['controls_report_sha256']
p=json.loads((pack/'report.json').read_text());assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3';c=json.loads((root/'controls.json').read_text())['records'];groups={}
for i in range(8064):
 for arm in ['quiet','noise15','rir','combined']:
  if arm in ['quiet','combined']:
   d=p['records'][3*i+(0 if arm=='quiet' else 2)];path=pack/d['raw'];expected=d['raw_file_sha256'];end=32000+d['signal_samples']
  else:
   d=c[2*i+(0 if arm=='noise15' else 1)];path=root/d['files']['raw']['path'];expected=d['files']['raw']['sha256'];end=32000+d['signal_samples']
  assert sha(path)==expected;raw=np.load(path,allow_pickle=False);assert raw.dtype==np.uint16 and raw.shape[1]==40
  frame_end=(np.arange(len(raw))+1)*160;signal=(frame_end>32000)&(frame_end<=end)
  for region,x in [('full',raw),('signal',raw[signal])]:
   key=arm+'/'+region;g=groups.setdefault(key,dict(arrays=0,values=0,zeros=0,clipped_above_26=0,max_raw=0,band_sum=np.zeros(40,np.float64),frames=0))
   g['arrays']+=1;g['values']+=x.size;g['zeros']+=int(np.count_nonzero(x==0));g['clipped_above_26']+=int(np.count_nonzero(x.astype(np.float32)/25.6>26));g['max_raw']=max(g['max_raw'],int(x.max()));g['band_sum']+=x.astype(np.float64).sum(axis=0);g['frames']+=len(x)
 if i%500==0:print(json.dumps(dict(sources=i+1,total=8064)),flush=True)
for g in groups.values():g['band_mean_raw']=(g.pop('band_sum')/g['frames']).tolist();g['clip_fraction']=g['clipped_above_26']/g['values'];g['zero_fraction']=g['zeros']/g['values']
r=dict(sources=8064,arrays=32256,groups=groups,script_sha256=sha(Path(__file__)),limits='Signal-region frame endpoints are generated-container references, not phonetic boundaries. Original frontend values before clip(raw/25.6,0,26);no alternate preprocessing/model scoring/training/newdev/final access or threshold selection.')
Path('research/wakeword_p4/results/4090-factorial-feature-range-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:{x:v for x,v in g.items() if x!='band_mean_raw'} for k,g in groups.items()}),flush=True)
