"""Independent full330000 raw-window reconstruction of saved auxiliary schedule."""
import gzip,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 result=R/'4090-engine-adversarial-schedule-result-v1.json';lock=R/'4090-engine-adversarial-schedule-execution-v1.json';r=json.loads(result.read_text());d=json.loads(lock.read_text());assert r['all_passed'] and r['execution_sha256']==sha(lock) and sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes']
 original=json.loads(Path(d['original_input_lock']).read_text());metadata=original['metadata'];binary=original['raw'];assert sha(metadata)==d['files'][metadata] and sha(binary)==d['files'][binary];rows=json.loads(Path(metadata).read_text());raw=np.memmap(binary,dtype='<u2',mode='r').reshape(-1,40);pools=defaultdict(list)
 for i,row in enumerate(rows):
  assert row['split']=='training'
  if row['source_pair_key']['carrier']=='bare' and row['source_pair_key']['scale']==1:pools[row['text'],row['engine'],row['variant']].append(i)
 keys=sorted({(t,e) for t,e,v in pools});texts=sorted({t for t,e in keys});engines=sorted({e for t,e in keys});assert len(keys)==33 and texts==r['texts'] and engines==r['engines']
 sr=np.random.default_rng(20261115);er=np.random.default_rng(20261116);pr=np.random.default_rng(20261117);counts=Counter();sources=defaultdict(set);conditions=defaultdict(set);original50=json.loads(Path(d['original_result']).read_text())['batches'];checked=0
 with gzip.open(r['path'],'rt') as f:
  for number,line in enumerate(f,1):
   rec=json.loads(line);variant=(number-1)%3;entries=[];xs=[];ey=[];ty=[];assert rec['step']==number and rec['variant']==variant
   for text,engine in keys:
    pool=pools[text,engine,variant];index=pool[int(sr.integers(0,len(pool)))];row=rows[index];boundary=math.ceil(row['generated_container_end_sample']/160)-2;lo=max(225,boundary+3);hi=min(row['frames'],boundary+27);end=int(er.integers(lo,hi+1));start=end-225;assert 0<=start and lo<=end<=hi and end<=row['frames'];at=row['offset'];crop=np.clip(np.array(raw[at+start:at+end],dtype=np.float32)/np.float32(25.6),0,26);value=np.concatenate([crop[1:],np.zeros((1,40),np.float32)]);xs.append(value);ey.append(engines.index(engine));ty.append(texts.index(text));entries.append(dict(row=index,end=end,legal_min=lo,legal_max=hi,source_id=row['source_id'],engine=engine,text=text,variant=variant));key=(engine,text,variant);counts[key]+=1;sources[key].add(row['source_id']);conditions[key].add(str(row['conditioning_id']));checked+=1
   order=pr.permutation(33);x=np.stack(xs)[order];engine_labels=np.array(ey,dtype=np.int32)[order];lexical_labels=np.array(ty,dtype=np.int32)[order]
   assert rec['entries']==[entries[i] for i in order] and len(rec['entries'])==33
   for values,key in [(x,'input_sha256'),(engine_labels,'engine_sha256'),(lexical_labels,'lexical_sha256')]:assert hashlib.sha256(values.tobytes()).hexdigest()==rec[key]
   assert np.bincount(engine_labels).tolist()==[11]*3 and np.bincount(lexical_labels).tolist()==[3]*11
   if number<=50:
    old=original50[number-1];assert rec['entries']==old['entries'] and rec['input_sha256']==old['input_sha256']
   if number%1000==0:print(json.dumps(dict(batches=number,independent_windows=checked)),flush=True)
 assert number==10000 and checked==330000 and len(counts)==99
 summary=[dict(engine=k[0],text=k[1],variant=k[2],draws=v,unique_source_metadata_ids=len(sources[k]),unique_conditioning_metadata_ids=len(conditions[k])) for k,v in sorted(counts.items())]
 for (e,t,v),n in counts.items():assert n==3334 if v==0 else n==3333
 out=dict(source_result_sha256=sha(result),execution_sha256=sha(lock),auditor_sha256=sha(__file__),all_passed=True,batches=10000,independent_actual_windows=checked,all_ordered_rows_endpoints_phase_tensor_and_label_hashes_exact=True,first50_original_receipts_exact=True,counts=summary,model_invocations=0,limits='Independent complete raw/metadata/RNG coordinate reconstruction, not model or formal training. Conditioning/source metadata IDs do not certify independent physical speakers; generated-end proxy is not phonetic truth. No freshdata or qualification.')
 with (R/'4090-engine-adversarial-schedule-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print('All10000 auxiliary batches/330000 windows independently reconstructed')
if __name__=='__main__':main()
