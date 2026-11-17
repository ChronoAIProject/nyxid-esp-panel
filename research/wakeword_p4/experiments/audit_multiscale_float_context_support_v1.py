"""Every saved whole-context container, both final arms and precisions."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');out={'schema':'multiscale-float-context-support-v1','arms':{},'input_sha256':{}}
for arm in ['control','multiscale']:
 p=Path(f'.local/wakeword/multiscale-float-v1/{arm}/result.json');out['input_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();report=json.loads(p.read_text());streams={}
 for r in report['rows']:
  if r['cohort']!='context':continue
  trace=r['trace'];p=Path(trace['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==trace['sha256'];out['input_sha256'][str(p)]=trace['sha256']
  with np.load(p) as z:clock=z['relative_block_return'].copy();values={'float':z['probability'].astype(np.float64),'int8':z['uint8'].astype(np.float64)/256}
  sums={k:np.convolve(v,np.ones(5),'full')[:len(v)]/5 for k,v in values.items()};rows=[]
  for t in r['timeline']:
   if t['kind']!='clip':continue
   mask=(clock>t['start'])&(clock<=t['end']);assert mask.any()
   row=dict(text_index=t['text_index'],text=t['text'],group=t['group'],precisions={})
   for k,v in sums.items():
    ev=[e for e in r['events'][k] if t['start']<e['relative_sample']<=t['end']]
    row['precisions'][k]=dict(maximum_five_mean=float(v[mask].max()),threshold=.91796875,above_threshold_calls=int(np.count_nonzero(v[mask]>=.91796875)),events=ev)
   rows.append(row)
  assert len(rows)==48;streams[r['index']]=rows
 assert len(streams)==3;out['arms'][arm]=streams
out['limits']='Consumed saved traces only; no new inference, data, exposure, tuning or P4 evidence. Means retain preceding physical context. Threshold crossing alone does not prove decoder eligibility. Container identity is not phonetic causation.'
(R/'4090-multiscale-float-context-support-v1.json').write_text(json.dumps(out,indent=2)+'\n')
for arm in out['arms']:
 print(arm,[(r['text_index'],[(k,round(v['maximum_five_mean'],6),len(v['events'])) for k,v in r['precisions'].items()]) for r in out['arms'][arm]['melo-BR'] if r['text_index'] in [2,8,9,10]])
