"""Two original seed726 batches, replacement identities and mask checks; no model."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-paired-carrier-preflight-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 base=R/'4090-binary40-multiscale-training-plan-v1.json';files=dict(json.loads(base.read_text())['files'])
 for p in [base,Path(__file__),Path(__file__).with_name('causal_paired_carrier_batch_v1.py'),Path(__file__).with_name('paired_carrier_proposal_v1.md'),R/'4090-multiscale-crossstore-carrier-support-v1.json']:files[str(p)]=sha(p)
 with LOCK.open('x') as f:json.dump({'files':files,'seed':726,'batches':2,'optimizer_updates':0},f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 from causal_source_balanced_batch_v1 import Sampler as Original
 from causal_paired_carrier_batch_v1 import Sampler
 a=Original(726);b=Sampler(726);records=[]
 for batch in range(2):
  old,od=a.batch();new,nd=b.batch();changed=[i for i,(x,y) in enumerate(zip(od,nd)) if x!=y];assert len(changed)==8
  assert sum(r['label'] for r in nd)==32;assert all(np.isfinite(v).all() for v in new)
  for i in range(128):
   if i not in changed:assert od[i]==nd[i] and all(np.array_equal(x[i],y[i]) for x,y in zip(old,new))
   assert np.isclose(new[1][i].sum(),nd[i]['label']) and np.isclose(new[2][i].sum(),1);assert not np.any((new[1][i]>0)&(new[2][i]>0))
  positives=[r for r in nd if r['kind']=='paired_carrier_positive'];negative=[r for r in nd if r['kind']=='paired_carrier_nick_negative'];assert len(positives)==len(negative)==4
  for x,y in zip(positives,negative):
   for k in ['conditioning_id','carrier','speed','variant','positive_offset','negative_offset']:assert x[k]==y[k]
  records.append({'batch':batch,'changed_slots':changed,'features_sha256':hashlib.sha256(new[0].tobytes()).hexdigest(),'positive_sha256':hashlib.sha256(new[1].tobytes()).hexdigest(),'negative_sha256':hashlib.sha256(new[2].tobytes()).hexdigest(),'pairs':positives})
 out={'schema':'paired-carrier-preflight-original-v1','all_passed':True,'plan_sha256':sha(LOCK),'batches':records,'unchanged_rows_verified':240,'optimizer_updates':0,'limits':'Sampler identity/geometry/masks only, no model gradient, training, recognition or promotion evidence.'};(R/'4090-paired-carrier-preflight-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
