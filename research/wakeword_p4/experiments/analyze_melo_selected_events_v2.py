"""Consumed-development diagnostics only; never select a rescue point."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('.local/wakeword/melo-development-int8-v2/metrics')
OUTPUT=Path('research/wakeword_p4/results/4090-melo-selected-event-diagnostic-v2.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(path):
 groups=defaultdict(lambda:dict(clips=0,activated=0,post_source_hits=0,events=0,before_generated_prefix=0,signed_first_prefix_ms=[]))
 with gzip.open(path,'rt') as f:
  rows=[json.loads(line) for line in f]
 assert len(rows)==474
 hits={}
 for data in rows:
  row=data['record']
  if row['phase']!='synthetic':continue
  events=[e for e in data['events'] if not e['priming']]
  hit=any(e['sample']>=row['lead_samples'] for e in events)
  hits[row['id']]=hit
  keys=['phrase/'+row['phrase_id'],'voice-phrase/'+row['voice_id'].split(':')[-1]+'/'+row['phrase_id'],'condition-phrase/'+str(row['variant'])+'/'+row['phrase_id'],'carrier-phrase/'+row['carrier']+'/'+row['phrase_id']]
  for key in keys:
   group=groups[key];group['clips']+=1;group['activated']+=int(bool(events));group['post_source_hits']+=int(hit);group['events']+=len(events)
   if events:
    latency=events[0]['generated_prefix_end_sample_signed_ms'];group['signed_first_prefix_ms'].append(latency);group['before_generated_prefix']+=int(latency<0)
 for group in groups.values():
  values=group.pop('signed_first_prefix_ms')
  group['first_event_generated_prefix_signed_ms']=dict(zip(['min','p25','median','p75','max'],map(float,np.percentile(values,[0,25,50,75,100])))) if values else None
 return dict(threshold=rows[0]['threshold'],groups=dict(groups)),hits
def main():
 models={};hits={};files={}
 for name in ['c161','speecht5-final20000','v30-frozen']:
  path=BASE/(name+'-event-details.jsonl.gz');files[str(path)]=sha(path);models[name],hits[name]=analyze(path)
 assert all(set(v)==set(hits['c161']) for v in hits.values())
 paired={}
 for name in ['c161','v30-frozen']:
  groups=defaultdict(lambda:dict(both_hit=0,new_only=0,reference_only=0,both_miss=0))
  for key,new in hits['speecht5-final20000'].items():
   ref=hits[name][key];field='both_hit' if new and ref else 'new_only' if new else 'reference_only' if ref else 'both_miss'
   phrase=next(r for r in ['nyx','official','nick','next','mix','muse'] if '-'+r+'-' in key)
   groups[phrase][field]+=1
  paired[name]=dict(groups)
 report=dict(schema='melo-selected-event-diagnostic-v2',models=models,paired=paired,files=files,selection_changed=False,acceptance_accessed=False,promotion_qualified=False,
  limits='Consumed-development descriptive diagnostics at the already selected per-model thresholds, not new threshold selection or validation. Internal synthesis prefix clocks are not acoustic phonetic truth. Aggregates include all actual post-prime events; post-source hits separately follow locked protocol. No causal attribution from before/after model changes or paired threshold changes alone.')
 with OUTPUT.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps({m:v['groups']['phrase/nyx'] for m,v in models.items()}))
if __name__=='__main__':main()
