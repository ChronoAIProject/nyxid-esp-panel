"""Full training metadata feasibility and independent Bernoulli KL math gate."""
import collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 material=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');refpath=R/'4090-causal-command-prefix-reference-map-v1.json'
 rows=[json.loads(x) for x in material.read_text().splitlines()];sources={r['source']['id']:r['source'] for r in rows};assert len(sources)==4320
 refdata=json.loads(refpath.read_text());assert refdata['mechanical_checks_passed'];refs={r['id']:r for r in refdata['records']}
 assert all(s['role']=='training' for s in sources.values()) and all(r['source_role']=='training' for r in rows)
 voices=sorted({(s['engine'],s['speaker_id']) for s in sources.values()});assert len(voices)==30
 negatives={v:sorted([s for s in sources.values() if (s['engine'],s['speaker_id'])==v and s['text']=='Hey Nick.'],key=lambda s:s['id'])[0] for v in voices}
 inventory=[];estimated=0
 for s in sorted(sources.values(),key=lambda s:s['id']):
  assert sha(s['native_path'])==s['native_sha256']
  context=negatives[(s['engine'],s['speaker_id'])];prefix=81600+((context['native_samples']+479)//480)*480+8160;assert prefix%480==0
  command=s['label']==1 and ',' in s['text']
  if command:assert s['id'] in refs and refs[s['id']]['native_sha256']==s['native_sha256']
  reference=refs[s['id']]['automatic_prefix_end_sample'] if command else s['native_samples'];assert 0<reference<=s['native_samples']
  frames=((prefix+s['native_samples']+16000+159)//160-2)//3*3;support=(3*np.arange(1,frames//3+1)+2)*160
  pos=(support>=prefix+reference+480)&(support<=prefix+reference+4320) if s['label'] else np.zeros(len(support),bool)
  neg=support<=prefix if s['label'] else np.ones(len(support),bool)
  transfer=(pos if s['label'] else neg)&(support>prefix)
  assert neg.any() and transfer.any() and (pos.any()==bool(s['label'])) and not np.any(pos&neg)
  inventory.append(dict(source=s,context=context['id'],prefix_samples=prefix,reference_samples=reference,frames=frames,positive_calls=int(pos.sum()),negative_calls=int(neg.sum()),transfer_calls=int(transfer.sum())))
  estimated+=2*frames*40*2
 # Complete immutable pairing schedule, separately replayed from a fresh RNG.
 pools={}
 for label in [0,1]:
  pools[label]={engine:{speaker:[i for i,r in enumerate(inventory) if r['source']['label']==label and r['source']['engine']==engine and r['source']['speaker_id']==speaker] for eng,speaker in voices if eng==engine} for engine in sorted({s['engine'] for s in sources.values()})}
 def schedule():
  rng=np.random.default_rng(237726);out=[]
  for step in range(1,20001):
   ids=[]
   for label in [1,1,0,0]:
    engines=sorted(pools[label]);ep=pools[label][engines[int(rng.integers(len(engines)))]];speakers=sorted(ep);pool=ep[speakers[int(rng.integers(len(speakers)))]];ids.append(pool[int(rng.integers(len(pool)))])
   out.append(dict(step=step,inventory_indices=ids))
  return out
 schedule_rows=schedule();assert schedule_rows==schedule();raw=('\n'.join(json.dumps(x,sort_keys=True) for x in schedule_rows)+'\n').encode();sp=R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz';sp.write_bytes(gzip.compress(raw,mtime=0))
 # Float64 stable KL with a detached independently computed target.
 checks=[]
 for teacher in [-100.,-20.,-1.,0.,1.,20.,100.]:
  for student in [-100.,-20.,-1.,0.,1.,20.,100.]:
   t=torch.tensor(teacher,dtype=torch.float64,requires_grad=True);z=torch.tensor(student,dtype=torch.float64,requires_grad=True);q=torch.sigmoid(t).detach();loss=torch.nn.functional.softplus(z)-q*z-(torch.nn.functional.softplus(t.detach())-q*t.detach());g,gt=torch.autograd.grad(loss,(z,t),allow_unused=True)
   expected=float(torch.sigmoid(z.detach())-q);assert gt is None and math.isfinite(float(loss)) and float(loss)>=-1e-12;assert abs(float(g)-expected)<1e-12
   def val(v):return max(v,0)+math.log1p(math.exp(-abs(v)))-float(q)*v
   h=1e-5;fd=(val(student+h)-val(student-h))/(2*h);assert abs(fd-expected)<1e-7
   if teacher==student:assert abs(float(loss))<1e-12 and abs(float(g))<1e-12
   checks.append(dict(teacher=teacher,student=student,kl=float(loss.detach()),gradient=float(g),finite_difference=fd))
 assert len(checks)==49
 result=dict(schema='frozen-history-teacher-metadata-math-preflight-v1',all_passed=True,files={str(material):sha(material),str(refpath):sha(refpath),__file__:sha(__file__)},sources=4320,voices=30,command_references=sum(r['source']['label']==1 and ',' in r['source']['text'] for r in inventory),label_counts=dict(collections.Counter(r['source']['label'] for r in inventory)),estimated_uint16_feature_bytes=estimated,inventory=inventory,schedule=dict(path=str(sp),sha256=sha(sp),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),steps=20000,pairs=80000,unique_sources=len({i for r in schedule_rows for i in r['inventory_indices']})),math_checks=checks,optimizer_updates=0,limits='Metadata/native SHA and deterministic source-schedule/float64 math only. No physical feature cache, actual teacher logits/gradients, one-step model wiring or efficacy gate passed by this report. Full4320teacher contradiction audit still required. No protected inputs or coefficient selection.')
 (R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['all_passed','sources','voices','command_references','label_counts','estimated_uint16_feature_bytes','schedule']}))
if __name__=='__main__':main()
