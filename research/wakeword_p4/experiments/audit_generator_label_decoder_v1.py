"""Training metadata oracle: allocation/decoder compatibility, not acoustic accuracy."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_overlap_targets_v1 import targets
from ordered_generator_matcher_v1 import OrderedMatcher
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-causal-prospective-training-index-v1.json.gz'
 with gzip.open(p,'rt') as f:data=json.load(f)
 records=[]
 for i,r in enumerate(data['records']):
  assert r['role']=='training'
  kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}
  y,ends,mass=targets(r['samples'],r['lead'],r['intervals'],**kwargs)
  dense_y=np.zeros((len(y),len(data['dense_to_original_label'])),dtype=y.dtype)
  for k,label in enumerate(data['dense_to_original_label']):
   if label<y.shape[1]:dense_y[:,k]=y[:,label]
  assert np.allclose(dense_y.sum(1),1,atol=1e-6)
  labels=dense_y.argmax(1);matcher=OrderedMatcher();events=[];collapsed=[];last=object()
  for j,dense in enumerate(labels):
   original=data['dense_to_original_label'][int(dense)];token=original-1 if original else None
   if token!=last:collapsed.append(token);last=token
   if token is None:matcher=OrderedMatcher()
   elif matcher.push(token):events.append(int(ends[j]))
  records.append(dict(index=i,class_id=r['class_id'],conditioning=r['conditioning'],carrier=r['carrier'],events_samples=events,collapsed_tokens=collapsed))
 summary={}
 for c in range(1,7):
  rows=[r for r in records if r['class_id']==c];summary[str(c)]=dict(records=len(rows),activated=sum(bool(r['events_samples']) for r in rows))
 out=dict(schema='generator-label-decoder-compatibility-v1',inputs={str(p):sha(p),__file__:sha(__file__),**{str(q):sha(q) for q in [Path('research/wakeword_p4/training/generator_overlap_targets_v1.py'),Path('research/wakeword_p4/training/ordered_generator_matcher_v1.py')]}},summary=summary,records=records,limits='Metadata-only soft-label argmax exact-matcher oracle on training partition. No PCM or trained-model calls. Not a bound on achievable recognition, acoustic truth, INT8 performance or fresh acceptance. Tie convention NumPy first index as dense training mapping.')
 dest=R/'4090-generator-label-decoder-compatibility-v1.json.gz'
 with dest.open('xb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as z:z.write((json.dumps(out,separators=(',',':'))+'\n').encode())
 print(json.dumps(summary))
if __name__=='__main__':main()
