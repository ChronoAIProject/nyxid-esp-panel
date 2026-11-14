"""Deterministic balanced training-only auxiliary batches, no models or main RNG."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class AuxiliaryBatches:
 def __init__(self):
  self.plan=json.loads((R/'4090-engine-adversarial-input-execution-v1.json').read_text());d=self.plan
  assert np.__version__==d['numpy']
  for p in [d['metadata'],d['raw']]:assert sha(p)==d['files'][p]
  self.rows=json.loads(Path(d['metadata']).read_text());self.raw=np.memmap(d['raw'],dtype='<u2',mode='r').reshape(-1,40);self.cells={}
  for i,r in enumerate(self.rows):
   assert r['split']=='training'
   if r['source_pair_key']['carrier']=='bare' and r['source_pair_key']['scale']==1.0:self.cells.setdefault((r['text'],r['engine'],r['variant']),[]).append(i)
  self.keys=sorted({(t,e) for t,e,v in self.cells});self.texts=sorted({t for t,e in self.keys});self.engines=sorted({e for t,e in self.keys});assert len(self.keys)==33 and len(self.texts)==11 and self.engines==['kokoro','piper','qwen']
  assert all((t,e,v) in self.cells for t,e in self.keys for v in range(3))
  self.source=np.random.default_rng(20261115);self.end=np.random.default_rng(20261116);self.permutation=np.random.default_rng(20261117);self.count=0
 def next(self):
  assert self.count<10000;variant=self.count%3;entries=[];values=[];el=[];tl=[]
  for text,engine in self.keys:
   pool=self.cells[text,engine,variant];i=int(pool[int(self.source.integers(len(pool)))]);r=self.rows[i];f=(r['generated_container_end_sample']+159)//160-2;lo=max(225,f+3);hi=min(r['frames'],f+27);assert lo<=hi;end=int(self.end.integers(lo,hi+1));at=r['offset'];crop=np.clip(np.asarray(self.raw[at+end-225:at+end],np.float32)/25.6,0,26);x=np.zeros_like(crop);x[:-1]=crop[1:];values.append(x);el.append(self.engines.index(engine));tl.append(self.texts.index(text));entries.append(dict(row=i,end=end,legal_min=lo,legal_max=hi,source_id=r['source_id'],engine=engine,text=text,variant=variant))
  order=self.permutation.permutation(33);x=np.stack(values)[order];ey=np.asarray(el,dtype=np.int32)[order];ty=np.asarray(tl,dtype=np.int32)[order];entries=[entries[j] for j in order];self.count+=1
  assert x.shape==(33,225,40) and np.isfinite(x).all() and np.bincount(ey).tolist()==[11]*3 and np.bincount(ty).tolist()==[3]*11
  receipt=dict(step=self.count,variant=variant,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),engine_sha256=hashlib.sha256(ey.tobytes()).hexdigest(),lexical_sha256=hashlib.sha256(ty.tobytes()).hexdigest(),entries=entries)
  return (x,ey,ty),receipt
