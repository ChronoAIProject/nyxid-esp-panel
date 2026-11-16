import gzip,hashlib,json,collections
from pathlib import Path
p=Path('research/wakeword_p4/results/4090-causal-unit-training-index-v1.json.gz');d=json.load(gzip.open(p));pat=[50,156,47,102,16,56,156,102,53,61];out={}
for cls in range(1,7):
 c=collections.Counter();dur=collections.Counter();
 for r in d['records']:
  if r['class_id']!=cls:continue
  ids=[x['token_id'] for x in r['intervals']];
  for i in range(len(ids)-1):
   if ids[i]==61:c[(ids[i+1], 'after61')]+=1;dur[ids[i]]+=r['intervals'][i]['end_sample']-r['intervals'][i]['start_sample']
  for i in range(len(ids)-1):
   if ids[i]==53:c[(ids[i+1],'after53')]+=1
 out[str(cls)]={'contexts':{str(k):v for k,v in c.items()},'terminal61_duration':dict(count=dur[61],minimum=None,maximum=None)}
 xs=[]
 for r in d['records']:
  if r['class_id']==cls:
   for i,x in enumerate(r['intervals'][:-1]):
    if x['token_id']==61:xs.append(x['end_sample']-x['start_sample'])
 if xs:out[str(cls)]['terminal61_duration']={'count':len(xs),'minimum':min(xs),'maximum':max(xs)}
res={'schema':'boundary-context-support-v1','index_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'classes':out,'limitations':'Metadata allocation only; no acoustic truth or model outputs. Context counts preserve class/carrier matched corpus and do not justify blanket token penalties.'}
dest=Path('research/wakeword_p4/results/4090-boundary-context-support-v1.json');dest.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(out,indent=2))
