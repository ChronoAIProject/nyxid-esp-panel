"""Authenticate original INT8 traces, summarize fixed precision, archive/reopen."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
import replay_binary40_float_v2 as old
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/shorter-span-float-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
out={'schema':'shorter-span-float-package-review-v1','arms':{},'limits':'Consumed synthetic diagnostics, no new exposure/tuning/P4. Historical role labels preserved. Background hours not additive.'}
for arm in ['control','shortspan']:
 old.introot=lambda c:Path(f'.local/wakeword/binary40-shorter-span-{arm}-{c}-replay-v1')
 old.intlock=lambda c:R/f'4090-binary40-shorter-span-{arm}-{c}-replay-plan-v1.json'
 rows=read(B/arm/'result.json')['rows'];lookup={(r['cohort'],r['phase'],r['index']):r for r in rows};bound=0
 for c in old.COHORTS:
  for i,s,pcm,path,h,q,orig,timeline in old.inputs(c):
   assert sha(path)==h;r=lookup[c,orig['phase'] if orig else 'synthetic',i];assert r['source']==s
   with np.load(path) as a,np.load(r['trace']['path']) as b:assert np.array_equal(a['uint8'],b['uint8'])
   bound+=1
 for r in [r for r in rows if r['cohort']=='context']:
  ref=next(s for s in read(R/f'4090-binary40-shorter-span-{arm}-context-original-v1.json')['streams'] if s['name']==r['index']);assert sha(ref['trace']['path'])==ref['trace']['sha256']
  with np.load(ref['trace']['path']) as a,np.load(r['trace']['path']) as b:assert np.array_equal(a['uint8'],b['uint8']) and np.array_equal(a['relative_block_return'],b['relative_block_return'])
  bound+=1
 assert bound==1763
 groups={}
 for r in rows:
  s=r['source'];c=r['cohort']
  if c=='context':
   for t in r['timeline']:
    if t['kind']!='clip':continue
    key='context/'+s['name']+'/'+t['group'];g=groups.setdefault(key,dict(rows=0,float=0,int8=0));g['rows']+=1
    for n in ['float','int8']:g[n]+=any(t['start']<e['relative_sample']<=t['end'] for e in r['events'][n])
   continue
  if r['phase']=='background':continue
  target=s.get('target_id')=='hey_nyx' or s.get('phrase_id')=='nyx' or s.get('target')=='hey_nyx' or s.get('group')=='nyx'
  if not target:continue
  key=c+('/'+s['voice_id'].split(':')[-1] if c=='melo' else '');g=groups.setdefault(key,dict(rows=0,float=0,int8=0));g['rows']+=1
  onset=s.get('lead_samples',0) if c in ['primed','jenny','joe'] else 0
  for n in ['float','int8']:g[n]+=any(e['relative_sample']>0 and e['relative_sample']>=onset for e in r['events'][n])
 paths=sorted((B/arm).glob('*.npz'))+[B/arm/'result.json',B/arm/'records.jsonl'];manifest={str(p.relative_to(B)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
 archive=R/f'4090-shorter-span-float-{arm}-evidence-v1.tar.gz'
 with tarfile.open(archive,'x:gz',compresslevel=6) as t:
  for p in paths:t.add(p,arcname=str(p.relative_to(B)),recursive=False)
 with tarfile.open(archive,'r:gz') as t:
  assert len(t.getmembers())==len(manifest)
  for m in t.getmembers():
   v=manifest[m.name];b=t.extractfile(m).read();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 out['arms'][arm]=dict(int8_original_streams_bound=bound,groups=groups,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),members=manifest)
 print(arm,json.dumps(groups),flush=True)
with (R/'4090-shorter-span-float-package-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
