"""Matched saved INT8 support census across rejected v40-v44; no new inference."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(pack/'records.json')=='757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
assert sha(pack/'sources.json')=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
records=json.loads((pack/'records.json').read_text());sources=json.loads((pack/'sources.json').read_text())
result={};inputs={}
for version in ['v40','v41','v42','v43','v44']:
 root=base/f'{version}-training-support-v1';report=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==report['scores_sha256']
 inputs[version]={'report_sha256':sha(root/'report.json'),'scores_sha256':report['scores_sha256'],'model_sha256':report['model_sha256']}
 groups={};historical=json.loads((base/f'eval-{version}-ami-extra-validation/report.json').read_text())['threshold']
 with np.load(root/'scores.npz') as z:
  for d in report['details']:
   i=d['pack_index'];r=records[i];alias=sources[r['source_index']]['aliases'][0]
   q=z[f'int8_{i}'];assert hashlib.sha256(q.tobytes()).hexdigest()==d['score_sha256']
   five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid')
   first=150+3*np.arange(len(five));last=first+12
   valid=((first-150)*160<=32000)&((last-150)*160<=32000)&((first+1)*160>=32000+r['signal_samples'])
   identity=json.dumps(sorted(alias['speakers']),ensure_ascii=False)
   keys=['all/'+str(d['label']),d['engine']+'/'+str(d['label']),d['engine']+'/'+str(d['label'])+'/'+d['phrase'],d['engine']+'/'+str(d['label'])+'/'+d['phrase']+'/'+identity]
   for key in keys:
    g=groups.setdefault(key,{'variants':0,'valid_positive_five_positions':0,'positive_zero_support':0,'thresholds':{}});g['variants']+=1
    if d['label']:g['valid_positive_five_positions']+=int(valid.sum());g['positive_zero_support']+=int(not valid.any())
    for t in sorted(set([.5,.992,.99609375,historical])):
     h=g['thresholds'].setdefault(str(t),dict(center_window_float=0,full_int8_five=0,positive_support_hit_variants=0,positive_support_hit_positions=0))
     h['center_window_float']+=int(d['window_float']>=t);h['full_int8_five']+=int(five.max()>=t)
     if d['label']:
      hits=int(np.count_nonzero(five[valid]>=t));h['positive_support_hit_variants']+=int(hits>0);h['positive_support_hit_positions']+=hits
 result[version]={'historical_threshold':historical,'groups':groups}
 assert groups['all/1']['variants']==4602 and groups['all/0']['variants']==1995
 assert groups['all/1']['valid_positive_five_positions']==103663 and groups['all/1']['positive_zero_support']==2
out={'scope':'All6597 eligible old-training variants per model, matched saved INT8 histories','inputs':inputs,'versions':result,'script_sha256':sha(Path(__file__)),'limits':'Container ledger only, not physical receptive field or phonetic truth. Center float vs full INT8 confounds quantization and phase. Checkpoints differ; no causal isolation. Original history thresholds and common references are descriptive, not selection. No new inference, optimizer, new/final data or promotion.'}
Path('research/wakeword_p4/results/4090-training-support-comparison-v4.json').write_text(json.dumps(out,indent=2)+'\n')
for v,d in result.items():print(v,json.dumps({k:d['groups'][k] for k in ['all/0','all/1']}))
