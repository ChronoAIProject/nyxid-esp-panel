"""Stratify all existing broad positive training sources; no scored-source selection."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
meta=json.loads(Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json').read_text())
byoffset={r['offset']:r for r in meta}
old=json.loads((R/'4090-binary40-broad-fit-original-v1.json').read_text())
out={'schema':'frozen-history-teacher-voice-retention-v1','arms':{},'limits':'All broad positive cached training rows, grouped by engine/conditioning; conditioning IDs are training voice identities, not independent human speakers or accent labels. No development voice matching or source selection. Other main sampler families remain unaudited for exhaustive fit.'}
for arm in ['control','teacher']:
 p=R/f'4090-frozen-history-teacher-{arm}-broad-fit-original-v1.json';new=json.loads(p.read_text());groups={}
 for before,after in zip(old['records'],new['records'],strict=True):
  assert before['source_id']==after['source_id'] and before['offset']==after['offset']
  if not after['label']:continue
  m=byoffset[after['offset']];key=m['engine']+'/'+m['conditioning_id'];g=groups.setdefault(key,dict(rows=0,original_hits=0,final_hits=0,gains=0,losses=0));a=bool(before['positive_crossings']);b=bool(after['positive_crossings']);g['rows']+=1;g['original_hits']+=a;g['final_hits']+=b;g['gains']+=b and not a;g['losses']+=a and not b
 assert len(groups)==716 and sum(v['rows'] for v in groups.values())==6444
 out['arms'][arm]=dict(report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),voices=groups,voices_with_losses=sum(v['losses']>0 for v in groups.values()),voices_with_gains=sum(v['gains']>0 for v in groups.values()),voices_net_worse=sum(v['final_hits']<v['original_hits'] for v in groups.values()),voices_net_better=sum(v['final_hits']>v['original_hits'] for v in groups.values()))
with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:
 first=json.loads(next(f))
out['original_first_batch_kinds']=dict(collections.Counter((d['kind']+'/'+str(d['label'])) for d in first['descriptors']))
(R/'4090-frozen-history-teacher-voice-retention-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({a:{k:v for k,v in d.items() if k!='voices'} for a,d in out['arms'].items()}));print(out['original_first_batch_kinds'])
