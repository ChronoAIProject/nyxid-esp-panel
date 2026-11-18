"""Independent temporal masks and paired original/candidate training fit transitions."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');meta=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((R/'4090-cross-engine-carrier-broad-fit-plan-v1.json').read_bytes());assert sha(meta)==lock['files'][str(meta)]
rows=[r for r in json.loads(meta.read_bytes()) if r['source_pair_key']['carrier']=='bare' and r['text'] in ('Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')]
reports=[R/'4090-binary40-broad-fit-original-v1.json',R/'4090-cross-engine-carrier-broad-fit-original-v1.json'];ds=[json.loads(p.read_bytes()) for p in reports];arrays=[]
for d in ds:
 assert sha(d['trace']['path'])==d['trace']['sha256']
 with np.load(d['trace']['path'],allow_pickle=False) as z:arrays.append(z['probability'].copy())
assert len(rows)==31644 and all(a.shape==(31644,282) for a in arrays)
groups={};changes=[]
for i,r in enumerate(rows):
 n=(r['frames']+2)//3;end=r.get('generated_container_end_sample');mask=[bool(r['label'] and end+480<=((j+1)*3+2)*160<=end+4320) for j in range(n)]
 crossing=[]
 for a,d in zip(arrays,ds,strict=True):
  rec=d['records'][i]
  assert all(rec[k]==r[k] for k in ['offset','source_id','variant','engine','conditioning_id','text','label']) and rec['index']==i and rec['valid_calls']==n
  found=[j for j in range(5,n+1) if all(mask[j-5:j]) and sum(float(v) for v in a[i,j-5:j])>=1175/256]
  assert found==rec['positive_crossings'],i
  crossing.append(bool(found))
 old,new=crossing;key=r['engine']+'/'+r['text'];g=groups.setdefault(key,dict(rows=0,retained=0,gained=0,lost=0,neither=0));g['rows']+=1;g['retained' if old and new else 'lost' if old else 'gained' if new else 'neither']+=1
 if old!=new:changes.append(dict(index=i,source_id=r['source_id'],offset=r['offset'],variant=r['variant'],engine=r['engine'],conditioning_id=r['conditioning_id'],old_crossing=old,new_crossing=new))
out=dict(all_passed=True,verified_rows=31644,sources={str(p):sha(p) for p in [meta,*reports]},groups=groups,changes=changes,limits='All positive five-call masks independently reconstructed from original frame/sample geometry; paired training-only cold-cache float support, not deployed INT8/generalization or causal attribution. No optimization or inference.')
with (R/'4090-cross-engine-bare-transitions-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(groups))
