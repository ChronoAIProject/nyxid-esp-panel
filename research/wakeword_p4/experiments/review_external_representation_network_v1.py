"""Saved-context/gradient audit, including unchanged historical supervised gradient."""
import gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');REPORT=R/'4090-external-representation-network-original-v1.json';LOCK=R/'4090-external-representation-network-review-plan-v1.json';E=Path('.local/wakeword/external-representation-network-execution-v1');OLD=R/'4090-frozen-history-teacher-network-traces-v1.npz'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 d=json.loads(REPORT.read_text());paths=[Path(__file__),REPORT,Path(d['trace']['path']),E/'terminal.json',E/'network.log',OLD,R/'4090-external-representation-target-cache-original-v1.json',Path('.local/wakeword/external-representation-target-cache-v1/targets.npz'),R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz']
 with LOCK.open('x') as f:json.dump(dict(files={str(p):sha(p) for p in paths},policy='All4320saved128contexts/source/anchor/head bindings, all50gradient vector reductions and analytic projection norms/losses, original50supervised gradient equality, actualexit/log/archive checks. No new model inference or claims to regenerate all intermediate backward tensors.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 d=json.loads(REPORT.read_text());terminal=json.loads((E/'terminal.json').read_text());assert terminal['actual_exit_code']==0 and terminal['log_sha256']==sha(E/'network.log') and d['student_projection_targets_unchanged'] and d['optimizer_updates']==0 and d['teacher_inference_calls']==0
 assert d['plan_sha256']==sha(R/'4090-external-representation-network-plan-v1.json') and sha(d['trace']['path'])==d['trace']['sha256']
 target_report=json.loads((R/'4090-external-representation-target-cache-original-v1.json').read_text())
 with np.load('.local/wakeword/external-representation-target-cache-v1/targets.npz',allow_pickle=False) as z:
  delta=z['raw_prefix'].astype(np.float64)-z['training_mean'];targets=(delta/np.linalg.norm(delta,axis=1,keepdims=True)).astype(np.float32)
 with gzip.open(R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz','rt') as f:schedule=[json.loads(s) for s in f]
 with np.load(d['trace']['path'],allow_pickle=False) as z,np.load(OLD,allow_pickle=False) as old:
  assert len(d['records'])==4320 and len(d['two_chunk_checks'])==30 and len(d['probes'])==50
  head=z['binary_head_weight'].reshape(-1).astype(np.float64);bias=float(z['binary_head_bias'][0]);pw=z['projection_weight'].astype(np.float64);pb=z['projection_bias'].astype(np.float64)
  for i,(r,t) in enumerate(zip(d['records'],target_report['records'],strict=True)):
   c=z[f'context_{i}'];logit=z[f'logit_{i}'];assert c.shape==(2,128) and np.isfinite(c).all() and r['index']==i and r['source_id']==t['source_id'] and r['anchor_zero_based_call']==t['student_anchor_zero_based_call']
   assert hashlib.sha256(c.tobytes()).hexdigest()==r['source_context_sha256'] and np.allclose(c.astype(np.float64)@head+bias,logit,atol=2e-4,rtol=2e-4)
  exact=0;max_old_difference=0.;stats=[]
  for r in d['probes']:
   step=r['step'];s=z[f'gS_{step}'].astype(np.float64);f=z[f'gD_{step}'].astype(np.float64);assert s.shape==f.shape==(30593,) and np.isfinite(s).all() and np.isfinite(f).all() and np.all(f[-129:]==0)
   previous=old[f'gS_{step}'];max_old_difference=max(max_old_difference,float(np.abs(previous-s).max()));exact+=int(np.array_equal(previous,z[f'gS_{step}']))
   sn=float(np.sqrt(s@s));fn=float(np.sqrt(f@f));dot=float(s@f);assert np.isclose(sn,r['supervised_norm'],rtol=1e-12,atol=1e-12) and np.isclose(fn,r['feature_student_norm'],rtol=1e-12,atol=1e-12) and np.isclose(dot,r['dot'],rtol=1e-12,atol=1e-12)
   ids=r['indices'];assert schedule[step-1]['step']==step and ids==schedule[step-1]['inventory_indices'];t=np.repeat(targets[ids],2,axis=0);assert hashlib.sha256(t.tobytes()).hexdigest()==r['target_sha256'] and np.array_equal(t.astype(np.float64),z[f'probe_target_{step}'])
   c=z[f'probe_context_{step}'];q=z[f'probe_q_{step}'];assert np.allclose(c.astype(np.float64)@pw.T+pb,q,atol=2e-4,rtol=2e-4)
   for j,i in enumerate(ids):assert np.allclose(c[2*j:2*j+2],z[f'context_{i}'],atol=2e-4,rtol=2e-4) and r['anchors'][2*j:2*j+2]==[target_report['records'][i]['student_anchor_zero_based_call']]*2
   t=t.astype(np.float64);qn=np.linalg.norm(q,axis=1,keepdims=True);tn=np.linalg.norm(t,axis=1,keepdims=True);cos=(q*t).sum(1,keepdims=True)/(qn*tn);assert np.isclose(float((1-cos).mean()),r['cosine_loss'],atol=2e-6,rtol=0)
   dq=(-t/(qn*tn)+(q*t).sum(1,keepdims=True)*q/(qn**3*tn))/len(q);wg=dq.T@c.astype(np.float64);bg=dq.sum(0);norm=float(np.sqrt(np.square(wg).sum()+np.square(bg).sum()));assert np.isclose(norm,r['feature_projection_norm'],atol=1e-6,rtol=1e-5)
   stats.append(dict(step=step,supervised_norm=sn,feature_student_norm=fn,ratio=fn/sn,dot=dot,cosine=dot/(sn*fn)))
  assert max_old_difference<1e-6
 archive=R/'4090-external-representation-network-evidence-v1.tar.gz';files=[Path(d['trace']['path']),REPORT,E/'terminal.json',E/'network.log'];members={}
 with tarfile.open(archive,'x:gz') as t:
  for f in files:members[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f));t.add(f,arcname=f.name)
 with tarfile.open(archive,'r:gz') as t:
  assert len(t.getnames())==len(members) and set(t.getnames())==set(members)
  for m in t:
   b=t.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
 out=dict(schema='external-representation-network-review-v1',all_passed=True,review_plan_sha256=sha(LOCK),report_sha256=sha(REPORT),source_pairs=4320,gradient_batches=50,historical_supervised_gradients_bit_identical=exact,historical_supervised_gradient_max_difference=max_old_difference,opposing_batches=sum(r['dot']<0 for r in stats),ratio_quantiles=np.quantile([r['ratio'] for r in stats],[0,.5,1]).tolist(),gradient_reductions=stats,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Independent saved anchor/head/projection/target/gradient reductions and original gradient comparison; graph/backward/immutability gates were original-runtime assertions, not an independent regeneration of intermediate tensors. No mature-trajectory, recognition or P4 claim.')
 with (R/'4090-external-representation-network-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['gradient_reductions','archive']}),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
