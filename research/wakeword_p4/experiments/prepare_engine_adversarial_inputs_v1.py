"""Training-only balanced engine preflight, no TensorFlow or models."""
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-adversarial-input-execution-v1.json';RESULT=R/'4090-engine-adversarial-input-result-v1.json';OUT=Path('.local/wakeword/engine-adversarial-input-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 if sys.argv[1]=='lock':
  census=R/'4090-hey-broad-surviving-window-census-execution-v2.json';old=json.loads(census.read_text());root=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx');meta=root/'features.json';raw=root/'features.bin';paths=[meta,raw,census,Path(__file__),Path(__file__).with_name('engine_adversarial_hypothesis_v1.md')]
  feature=R/'4090-hey-broad-features-v1.json';audit=R/'4090-hey-broad-feature-independent-audit-v1.json';fr=json.loads(feature.read_text());fa=json.loads(audit.read_text())
  assert fa['binary_hash_verified'] and fa['original_report_sha256']==sha(feature)
  assert sha(meta)==old['files'][str(meta)] and sha(raw)==fr['shared_binary_sha256'] and raw.stat().st_size==fr['shared_binary_bytes']
  paths.extend([feature,audit,Path(__file__).with_name('audit_engine_adversarial_inputs_v1.py'),R/'4090-engine-adversarial-upstream-evidence-v1.json'])
  files={str(p):sha(p) for p in paths}
  with LOCK.open('x') as f:json.dump(dict(files=files,metadata=str(meta),raw=str(raw),batches=50,rows_per_batch=33,seeds=dict(source=20261115,end=20261116,permutation=20261117),numpy=np.__version__,policy='Balanced33cells first50 training-only batches, no network/optimizer/newsource/freshdata. Full generated-container proxy endpoint, phase224past+zero. Every scheduled draw retained.'),f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 assert np.__version__==d['numpy'];rows=json.loads(Path(d['metadata']).read_text());raw=np.memmap(d['raw'],dtype='<u2',mode='r').reshape(-1,40)
 cells={}
 for i,r in enumerate(rows):
  assert r['split']=='training'
  if r['source_pair_key']['carrier']=='bare' and r['source_pair_key']['scale']==1.0:cells.setdefault((r['text'],r['engine'],r['variant']),[]).append(i)
 keys=sorted({(t,e) for t,e,v in cells});texts=sorted({t for t,e in keys});engines=sorted({e for t,e in keys});assert len(keys)==33 and len(texts)==11 and engines==['kokoro','piper','qwen']
 assert all((t,e,v) in cells for t,e in keys for v in range(3))
 for ids in cells.values():
  for i in ids:
   r=rows[i];f=(r['generated_container_end_sample']+159)//160-2;assert max(225,f+3)<=min(r['frames'],f+27)
 OUT.mkdir(exist_ok=False);rs=np.random.default_rng(d['seeds']['source']);re=np.random.default_rng(d['seeds']['end']);rp=np.random.default_rng(d['seeds']['permutation']);batches=[]
 try:
  for step in range(50):
   variant=step%3;entries=[];values=[];el=[];tl=[]
   for text,engine in keys:
    pool=cells[text,engine,variant];i=int(pool[int(rs.integers(len(pool)))]);r=rows[i];f=(r['generated_container_end_sample']+159)//160-2;lo=max(225,f+3);hi=min(r['frames'],f+27);end=int(re.integers(lo,hi+1));at=r['offset'];x=np.zeros((225,40),np.float32);x[:-1]=np.minimum(np.asarray(raw[at+end-224:at+end],np.float32)/25.6,26)
    assert x.shape==(225,40) and np.isfinite(x).all();values.append(x);el.append(engines.index(engine));tl.append(texts.index(text));entries.append(dict(row=i,end=end,legal_min=lo,legal_max=hi,source_id=r['source_id'],engine=engine,text=text,variant=variant))
   permutation=rp.permutation(33);x=np.asarray(values)[permutation];ey=np.asarray(el,dtype=np.int32)[permutation];ty=np.asarray(tl,dtype=np.int32)[permutation];entries=[entries[j] for j in permutation]
   assert np.bincount(ey).tolist()==[11]*3 and np.bincount(ty).tolist()==[3]*11
   p=OUT/f'{step+1:03}.npz';np.savez_compressed(p,x=x,engine=ey,lexical=ty)
   batches.append(dict(step=step+1,variant=variant,arrays_path=str(p),arrays_sha256=sha(p),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),entries=entries))
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),engines=engines,texts=texts,cell_counts=[dict(text=t,engine=e,variant=v,rows=len(ids),sources=len({rows[i]['source_id'] for i in ids})) for (t,e,v),ids in sorted(cells.items())],batches=batches,all_passed=True,model_invocations=0,limits=d['policy']),f,indent=2)
  print(json.dumps(dict(batches=len(batches),rows=1650,all_passed=True)))
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
