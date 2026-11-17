"""Census original synthetic non-target identities; do not infer natural prevalence."""
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
R=Path('research/wakeword_p4/results');p=R/'4090-c161-binary40-parallel-events-original-v2.json';raw=p.read_bytes();d=json.loads(raw)
lock=json.loads((R/'4090-c161-binary40-parallel-runtime-plan-v1.json').read_text());assert lock['files'][str(p)]==hashlib.sha256(raw).hexdigest()
assert json.loads((R/'4090-c161-binary40-parallel-runtime-review-v1.json').read_text())['all_passed']
models=['c161','binary40','parallel'];groups=defaultdict(Counter);records=[];background=[];continuous=[];total=0
for c,v in d['cohorts'].items():
 for r in v['records']:
  total+=1;s=r['source']
  if r['phase']=='background':
   background.append(dict(cohort=c,index=r['index'],source=s,events=dict(c161=r['saved']['c161'],binary40=r['saved']['binary40'],parallel=r['emitted'])));continue
  if c=='continuous':
   continuous.append(dict(index=r['index'],containers=r['containers'],outside_clip_events=r['outside_clip_events']));continue
  label=s.get('target_id',s.get('target',s.get('phrase_id',s.get('group'))))
  assert label is not None,(c,r['index'])
  if label in ['hey_nyx','nyx']:continue
  text=s['text'];normalized=re.sub(r'[^a-z0-9 ]','',text.lower()).strip()
  if c=='daily':category='daily:'+s['group']
  elif label in ['common_negative','confusion']:
   # Explicit literal labels only. These address carriers are constructed diagnostics.
   prefixes=['hey william','hey pillow','hey nick','hey next','hey muse','hey mix','hey nexus','hey fix','hey nice','hello','high speed','hi e s b','e s p','nicks','hey','hi']
   category=next((x for x in prefixes if normalized==x or normalized.startswith(x+' ')),None)
   assert category is not None,text
  else:category=label
  key=(c,category);g=groups[key];g['clips']+=1
  ev=dict(c161=r['saved']['c161'],binary40=r['saved']['binary40'],parallel=r['emitted']);hits={m:any(e['relative_sample']>=0 for e in ev[m]) for m in models}
  for m in models:g[m+'_hit_clips']+=hits[m];g[m+'_real_events']+=sum(e['relative_sample']>=0 for e in ev[m]);g[m+'_prime_events']+=sum(e['relative_sample']<0 for e in ev[m])
  g['added_parallel_hit_vs_c161']+=hits['parallel'] and not hits['c161'];g['removed_parallel_hit_vs_c161']+=hits['c161'] and not hits['parallel']
  records.append(dict(cohort=c,index=r['index'],category=category,original_label=label,text=text,source=s,hits=hits,events=ev))
assert total==1760
out=dict(schema='c161-binary40-non-target-original-v1',source_sha256=hashlib.sha256(raw).hexdigest(),rows=total,synthetic_non_target_clips=len(records),groups=[dict(cohort=c,category=k,**v) for (c,k),v in sorted(groups.items())],records=records,background=background,continuous=continuous,limits='Original labels/literal text only, no ASR or phonetic relabeling. Daily96 clips distinct from constructed address/nearword diagnostics. Keep cohort/voice identities; reused sources/variants are correlated, not new independent trials or population FAR. Background exposure unchanged and nonadditive. No inference/training/threshold or policy changes.')
with (R/'4090-c161-binary40-non-target-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print('rows',total,'non-target',len(records))
for g in out['groups']:
 if g['parallel_hit_clips']:print(json.dumps({k:v for k,v in g.items() if k.endswith('hit_clips') or k in ['cohort','category','clips','added_parallel_hit_vs_c161','removed_parallel_hit_vs_c161']}))
