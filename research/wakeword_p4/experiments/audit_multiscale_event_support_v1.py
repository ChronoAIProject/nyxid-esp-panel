"""Saved score and event identities only; no model inference or tuning."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results'); bindings={};out={'schema':'multiscale-event-support-v1','contexts':{},'background':{}}
def read(p):
 p=Path(p);b=p.read_bytes();bindings[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
for arm in ['control','multiscale']:
 s=read(R/f'4090-binary40-multiscale-{arm}-context-original-v1.json');out['contexts'][arm]={}
 for stream in s['streams']:
  t=stream['trace'];p=Path(t['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==t['sha256'];bindings[str(p)]=t['sha256']
  with np.load(p) as z:u=z['uint8'].astype(np.int64);clock=z['relative_block_return'].copy()
  sums=np.convolve(u,np.ones(5,dtype=np.int64),'full')[:len(u)]
  rows=[]
  for clip in stream['timeline']:
   if clip['kind']!='clip':continue
   mask=(clock>clip['start'])&(clock<=clip['end']); idx=np.flatnonzero(mask);assert len(idx)
   best=int(idx[np.argmax(sums[idx])]);events=[e for e in stream['events'] if clip['start']<e['relative_sample']<=clip['end']]
   rows.append({'text_index':clip['text_index'],'group':clip['group'],'text':clip['text'],'maximum_five_code_sum':int(sums[best]),'threshold_sum':1175,'best_step':best,'best_relative_sample':int(clock[best]),'threshold_crossing_calls':int(np.count_nonzero(sums[idx]>=1175)),'events':events})
  out['contexts'][arm][stream['name']]=rows
 s=read(R/f'4090-binary40-multiscale-{arm}-melo-comparison-v1.json')
 out['background'][arm]=[r for r in s['records'] if r['phase']=='background' and r['events']['binary40']]
out['input_sha256']=bindings
out['limits']='Consumed saved INT8 trace census only. Window sums include preceding physical context. No threshold tuning, new inference/exposure, phonetic attribution or P4 claim. A subthreshold maximum proves no decoder could emit within this container under fixed policy; a crossing alone does not establish decoder eligibility.'
(R/'4090-binary40-multiscale-event-support-v1.json').write_text(json.dumps(out,indent=2)+'\n')
for arm in out['contexts']:
 print(arm,'BR',[(r['text_index'],r['maximum_five_code_sum'],len(r['events'])) for r in out['contexts'][arm]['melo-BR'] if r['text_index'] in [2,8,9,10]])
 print(arm,'Max',[(r['maximum_five_code_sum'],r['events']) for r in out['contexts'][arm]['melo-US'] if r['text_index']==46])
