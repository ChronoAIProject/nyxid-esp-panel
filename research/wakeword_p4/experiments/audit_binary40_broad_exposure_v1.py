"""Count all saved training draws against exhaustive fixed broad-fit records."""
import collections,gzip,json
from pathlib import Path
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');T=Path('.local/wakeword/binary40-training-v1')
def main():
 fitpath=R/'4090-binary40-broad-fit-original-v1.json';fit=json.loads(fitpath.read_text());training=json.loads((R/'4090-binary40-training-original-v1.json').read_text());log=T/'batches.jsonl.gz'
 assert sha(log)==training['files']['batches.jsonl.gz']
 counts=collections.Counter();steps=0;bykind=collections.Counter()
 with gzip.open(log,'rt') as f:
  for line in f:
   r=json.loads(line);steps+=1;assert r['step']==steps and len(r['descriptors'])==128
   for d in r['descriptors']:
    bykind[d['kind']]+=1
    if d['kind'] in ['existing_broad_bare_positive','source_balanced_bare_negative']:counts[d['index']]+=1
 assert steps==20000
 rows=[];groups={}
 def bucket(n):return '0' if n==0 else '1-4' if n<5 else '5-9' if n<10 else '10+'
 for r in fit['records']:
  n=counts[r['offset']];failure=not bool(r['positive_crossings']) if r['label'] else any(s<=r['valid_calls'] for s in r['events'])
  group=r['engine']+'/'+r['text'];g=groups.setdefault(group,dict(rows=0,draws=0,zero_draw_rows=0,failure_rows=0,failure_zero_draw_rows=0,by_exposure={}))
  g['rows']+=1;g['draws']+=n;g['zero_draw_rows']+=n==0;g['failure_rows']+=failure;g['failure_zero_draw_rows']+=failure and n==0
  b=g['by_exposure'].setdefault(bucket(n),dict(rows=0,failure_rows=0));b['rows']+=1;b['failure_rows']+=failure
  rows.append(dict(index=r['index'],offset=r['offset'],draws=n,failure=bool(failure)))
 assert sum(g['draws'] for g in groups.values())==320000
 out=dict(schema='binary40-broad-exposure-v1',source_sha256={str(p):sha(p) for p in [Path(__file__),fitpath,log,R/'4090-binary40-training-original-v1.json']},steps=steps,all_descriptor_rows=steps*128,kind_counts=dict(bykind),groups=groups,records=rows,policy='Read every original batch once. Counts correspond to exact rendered feature offset, no inference/optimization/dev/acceptance. Positive failure = no complete supervised fivecall crossing; negative failure = event within valid support. Fixed descriptive exposure buckets0/1-4/5-9/10+; membership/exposure not independent generalization or causal proof. No reweighting selected.')
 p=R/'4090-binary40-broad-exposure-original-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(groups))
if __name__=='__main__':main()
