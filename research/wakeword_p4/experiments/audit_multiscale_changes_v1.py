"""Saved-result census only; no inference, tuning or new exposure."""
import collections
import hashlib
import json
from pathlib import Path
R=Path('research/wakeword_p4/results')
bindings={}
def read(name):
 p=R/name;b=p.read_bytes();bindings[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
out={'schema':'multiscale-source-change-census-v1','cohorts':{},'contexts':{},'limits':'Consumed synthetic saved-result census only. Historical source role labels are preserved, not current unconsumed status. No new inference, data, tuning, exposure or P4 claim.'}
for c in ['builtin','daily','primed','jenny','joe','melo','mac']:
 a,b=[read(f'4090-binary40-multiscale-{arm}-{c}-comparison-v1.json') for arm in ['control','multiscale']]
 assert len(a['records'])==len(b['records'])
 changes=[];counts=collections.Counter()
 for x,y in zip(a['records'],b['records']):
  assert x['index']==y['index']
  src=x.get('source',x.get('request'));assert src==y.get('source',y.get('request'))
  if x.get('hits',True) is None:
   assert y.get('hits') is None
   continue
  def hit(z):
   if 'hits' in z:return z['hits']['binary40']
   return bool(z.get('real_events',z.get('events')))
  before,after=hit(x),hit(y)
  if before==after:continue
  target=(src.get('target_id')=='hey_nyx' or src.get('target') in ['hey_nyx','hey_nicks'] or src.get('phrase_id')=='nyx' or src.get('group')=='nyx')
  direction='gained' if after else 'lost'
  key=('target' if target else 'non-target')+'/'+direction
  counts[key]+=1
  changes.append({'index':x['index'],'classification':key,'source':src,'before':before,'after':after})
 out['cohorts'][c]={'records_verified':len(a['records']),'counts':dict(counts),'changes':changes}
a,b=[read(f'4090-binary40-multiscale-{arm}-context-original-v1.json') for arm in ['control','multiscale']]
for x,y in zip(a['streams'],b['streams']):
 assert x['name']==y['name'] and x['timeline']==y['timeline']
 def owners(s):
  return {i for i,t in enumerate(s['timeline']) if t['kind']=='clip' and any(t['start']<e['relative_sample']<=t['end'] for e in s['events'])}
 before,after=owners(x),owners(y)
 out['contexts'][x['name']]={'changes':[{'timeline_index':i,'direction':'gained' if i in after else 'lost','container':x['timeline'][i]} for i in sorted(before^after)]}
out['input_sha256']=bindings
p=R/'4090-binary40-multiscale-source-changes-v1.json';p.write_text(json.dumps(out,indent=2)+'\n')
for k,v in out['cohorts'].items():print(k,v['counts'])
for k,v in out['contexts'].items():print(k,[(x['direction'],x['container']['text_index']) for x in v['changes']])
