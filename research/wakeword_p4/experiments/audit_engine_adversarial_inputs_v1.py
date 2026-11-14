"""Independent saved-coordinate/raw-tensor audit, no producer or model imports."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 lock=R/'4090-engine-adversarial-input-execution-v1.json';result=R/'4090-engine-adversarial-input-result-v1.json';d=json.loads(lock.read_text());r=json.loads(result.read_text());assert r['all_passed'] and r['execution_sha256']==sha(lock) and len(r['batches'])==50
 for p,h in d['files'].items():assert sha(p)==h,p
 rows=json.loads(Path(d['metadata']).read_text());raw=np.memmap(d['raw'],dtype='<u2',mode='r').reshape(-1,40);pools=defaultdict(list)
 for index,row in enumerate(rows):
  assert row['split']=='training'
  if row['source_pair_key']['carrier']=='bare' and row['source_pair_key']['scale']==1.0:pools[row['text'],row['engine'],row['variant']].append(index)
 keys=sorted({(t,e) for t,e,v in pools});assert r['engines']==sorted({e for t,e in keys}) and r['texts']==sorted({t for t,e in keys})
 assert r['cell_counts']==[dict(text=t,engine=e,variant=v,rows=len(ids),sources=len({rows[i]['source_id'] for i in ids})) for (t,e,v),ids in sorted(pools.items())]
 sr=np.random.default_rng(20261115);er=np.random.default_rng(20261116);pr=np.random.default_rng(20261117);count=0
 for batch in r['batches']:
  step=batch['step'];variant=(step-1)%3;expected=[];xs=[];ey=[];ty=[]
  for text,engine in keys:
   pool=pools[text,engine,variant];index=pool[int(sr.integers(0,len(pool)))];row=rows[index];boundary=int(np.ceil(row['generated_container_end_sample']/160))-2;lo=max(225,boundary+3);hi=min(row['frames'],boundary+27);end=int(er.integers(lo,hi+1));offset=row['offset'];crop=np.clip(np.asarray(raw[offset+end-225:offset+end],dtype=np.float32)/25.6,0,26);x=np.zeros_like(crop);x[:-1]=crop[1:]
   expected.append(dict(row=index,end=end,legal_min=lo,legal_max=hi,source_id=row['source_id'],engine=engine,text=text,variant=variant));xs.append(x);ey.append(r['engines'].index(engine));ty.append(r['texts'].index(text))
  order=pr.permutation(33);assert batch['entries']==[expected[i] for i in order] and batch['variant']==variant
  assert sha(batch['arrays_path'])==batch['arrays_sha256']
  with np.load(batch['arrays_path'],allow_pickle=False) as a:
   assert set(a.files)=={'x','engine','lexical'} and a['x'].dtype==np.float32 and a['x'].shape==(33,225,40)
   assert np.array_equal(a['x'],np.stack(xs)[order]) and np.array_equal(a['engine'],np.array(ey,dtype=np.int32)[order]) and np.array_equal(a['lexical'],np.array(ty,dtype=np.int32)[order])
   assert np.bincount(a['engine']).tolist()==[11]*3 and np.bincount(a['lexical']).tolist()==[3]*11
   assert hashlib.sha256(a['x'].tobytes()).hexdigest()==batch['input_sha256'];count+=33
 assert count==1650
 out=dict(execution_sha256=sha(lock),original_result_sha256=sha(result),source_sha256=sha(__file__),all_passed=True,batches=50,actual_windows=1650,all_rng_coordinates_tensors_exact=True,all_lexical_engine_variant_cells_balanced=True,model_invocations=0,limits='Training-only input feasibility; no network/gradient/optimizer or acoustic phonetic truth. Does not qualify old source data or open freshdata.')
 with (R/'4090-engine-adversarial-input-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
