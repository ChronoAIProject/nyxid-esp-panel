"""All360 independent integer cumulative-sum metrics and source bindings."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 P=R/'4090-binary40-long-history-factorial-plan-v1.json';Q=R/'4090-binary40-long-history-factorial-original-v1.json';p=json.loads(P.read_text());q=json.loads(Q.read_text());old=json.loads((R/'4090-binary40-context-int8-original-v1.json').read_text())
 assert q['plan_sha256']==sha(P) and len(q['records'])==len(old['records'])==len(p['sources'])==90 and q['all180_diagonal_byte_checks_passed']
 for path,h in p['files'].items():assert sha(path)==h,path
 assert sha(q['trace']['path'])==q['trace']['sha256'];arrays=np.load(q['trace']['path'],allow_pickle=False);assert len(arrays.files)==360
 counts=defaultdict(lambda:defaultdict(Counter));details=[]
 for i,(row,prior) in enumerate(zip(q['records'],old['records'],strict=True)):
  assert row['index']==prior['index']==i and row['source']==prior['source']==p['sources'][i]
  source=row['source'];prefix=81600+16*((source['context_negative']['native_samples']+479)//480*480+8160);assert row['prefix_samples']==prefix and row['split_feature']==((prefix//160-2)//3)*3
  assert sha(row['feature_file']['path'])==row['feature_file']['sha256'];fronts=np.load(row['feature_file']['path'],allow_pickle=False)
  samples=prefix+source['native_samples']+16000
  for name in ['silence','speech']:
   raw=fronts[name];assert raw.dtype==np.uint16 and raw.shape==((samples+159)//160-2,40) and hashlib.sha256(raw.tobytes()).hexdigest()==row['raw_frontend_sha256'][name]
  for front in ['silence','speech']:
   for cache in ['silence','speech']:
    key=front+'/'+cache;v=arrays[f'{i}_{front}_{cache}'];assert v.dtype==np.uint8
    sums=np.r_[0,np.cumsum(v,dtype=np.int64)];five=sums[5:]-sums[:-5];supports=(3*(np.arange(len(v))+row['split_feature']//3+1)+2)*160
    valid=(supports[4:]>=prefix)&(supports[4:]<=prefix+source['prefix_reference_samples']+4320);metric=dict(max_sum5=int(five[valid].max()),crossing=bool(np.any(five[valid]>=1175)));assert metric==row['arms'][key]
    before=prior['arms'][key]['crossing'];after=metric['crossing'];c=counts[source['text']][key];c['sources']+=1;c['single_crossings']+=before;c['long_crossings']+=after;c['gains']+=after and not before;c['losses']+=before and not after
    details.append(dict(index=i,source_id=source['id'],text=source['text'],arm=key,single=prior['arms'][key],long=metric))
 result=dict(schema='binary40-long-history-factorial-review-v1',all_passed=True,original_sha256=sha(Q),source_rows=90,metrics=360,physical_diagonal_checks=180,counts=counts,history_seconds=dict(min=min(r['history_seconds'] for r in q['records']),max=max(r['history_seconds'] for r in q['records'])),details=details,limits='Training-only repetition stressor, original support-ledger five-code crossing metric, not full decoder or natural conversation. Four offdiagonal arms are artificial and not deployed resets. No inference, fitting or parameter/threshold selection in review; not human/P4/independent accuracy.')
 with (R/'4090-binary40-long-history-factorial-review-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
