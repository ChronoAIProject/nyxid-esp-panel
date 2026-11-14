"""Independent complete saved training audit, no TF/optimizer/model invocation."""
import gzip,hashlib,json,sys
from pathlib import Path
import h5py
import numpy as np
R=Path('research/wakeword_p4/results');RESULT=R/'4090-engine-adversarial-formal-result-v1.json';LOCK=R/'4090-engine-adversarial-formal-execution-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def arrays(p):
 out={}
 with h5py.File(p) as f:
  def visit(k,v):
   if isinstance(v,h5py.Dataset):out[k]=v[()]
  f.visititems(visit)
 return out
def check_inventory(a,records):
 assert set(a.files)=={f'v{i}' for i in range(len(records))}
 for rec in records:
  v=a[f"v{rec['index']}"];assert list(v.shape)==rec['shape'] and str(v.dtype)==rec['dtype'] and np.isfinite(v).all() and hashlib.sha256(v.tobytes()).hexdigest()==rec['sha256']
def main():
 if not RESULT.exists():print(json.dumps(dict(status='not_ready',missing=str(RESULT),model_invocations=0)));return 2
 r=json.loads(RESULT.read_text());d=json.loads(LOCK.read_text());assert r['all_passed'] and r['execution_sha256']==sha(LOCK) and len(r['cases'])==4
 assert {(c['target'],c['condition']) for c in r['cases']}=={(t,c) for t in ['hey_nyx','hey_willow'] for c in ['control','adversarial']}
 assert sha(d['initial']['path'])==d['initial']['sha256'];original=arrays(d['initial']['path']);bn=[k for k in original if 'batch_normalization' in k];assert len(bn)==16;smoke=json.loads(Path(d['smoke_result']).read_text());old=json.loads(Path(d['old_formal']).read_text());checks=[];matched={}
 for case in r['cases']:
  target=case['target'];condition=case['condition'];folder=Path(case['folder']);assert case['steps']==case['optimizer_iterations']==10000 and case['coefficient']==d['coefficient'][condition] and case['bn_unchanged'];assert json.loads((folder/'training.json').read_text())==case
  for name,h in case['files'].items():assert sha(folder/name)==h
  assert sha(folder/'history.jsonl')==case['history_sha256'];init=arrays(folder/'initial.weights.h5');final=arrays(folder/'final.weights.h5');assert init.keys()==final.keys()==original.keys();assert all(np.array_equal(init[k],original[k]) for k in original) and all(np.isfinite(x).all() for x in final.values());assert all(np.array_equal(final[k],original[k]) for k in bn)
  fresh=next(c for c in smoke['cases'] if c['target']==target and c['condition']==condition)['repeats'][0];sourcefolder=Path(fresh['folder']);baseline=next(c for c in old['cases'] if c['target']==target and c['condition']=='full')
  with np.load(folder/'initial-state.npz',allow_pickle=False) as a,np.load(folder/'final-state.npz',allow_pickle=False) as b,np.load(sourcefolder/'initial-state.npz',allow_pickle=False) as source,np.load(folder/'optimizer.npz',allow_pickle=False) as opt:
   check_inventory(a,case['initial_inventory']);check_inventory(b,case['final_inventory']);check_inventory(opt,case['optimizer_inventory']);assert a.files==source.files and all(np.array_equal(a[k],source[k]) for k in a.files);assert len(opt.files)==36 and int(opt['v0'])==10000 and float(opt['v1'])==float(np.float32(1e-5));assert hashlib.sha256(b''.join(opt[f'v{i}'].tobytes() for i in range(36))).hexdigest()==case['optimizer_signature'];assert hashlib.sha256(b''.join(b[f'v{i}'].tobytes() for i in range(len(b.files)))).hexdigest()==case['state_signature']
   bn_indices=[rec['index'] for rec in case['initial_inventory'] if 'batch_normalization' in rec['name']];assert len(bn_indices)==16
   assert all(np.array_equal(a[f'v{i}'],b[f'v{i}']) for i in bn_indices)
   assert len(case['snapshots'])==10 and [s['step'] for s in case['snapshots']]==list(range(1000,10001,1000))
   for snapshot in case['snapshots']:
    step=snapshot['step'];prefix=f'step{step}'
    for name,h in snapshot['files'].items():assert sha(folder/name)==h
    saved=arrays(folder/(prefix+'.weights.h5'));assert saved.keys()==original.keys() and all(np.isfinite(v).all() for v in saved.values()) and all(np.array_equal(saved[k],original[k]) for k in bn)
    with np.load(folder/(prefix+'-state.npz'),allow_pickle=False) as state,np.load(folder/(prefix+'-optimizer.npz'),allow_pickle=False) as so:
     assert state.files==a.files and so.files==opt.files and int(so['v0'])==step and float(so['v1'])==float(np.float32(1e-5));assert all(np.isfinite(v).all() for p in [state,so] for v in p.values());assert all(np.array_equal(state[f'v{i}'],a[f'v{i}']) for i in bn_indices)
     for rec in case['initial_inventory']:assert state[f"v{rec['index']}"].shape==tuple(rec['shape']) and str(state[f"v{rec['index']}"].dtype)==rec['dtype']
     if step==10000:assert all(np.array_equal(saved[k],final[k]) for k in final) and all(np.array_equal(state[k],b[k]) for k in b.files) and all(np.array_equal(so[k],opt[k]) for k in opt.files)
   if condition=='control':
    assert sha(baseline['weights_path'])==baseline['weights_sha256'] and sha(baseline['optimizer_path'])==baseline['optimizer_sha256'];expected=arrays(baseline['weights_path']);assert all(np.array_equal(expected[k],final[k]) for k in final)
    with np.load(baseline['optimizer_path'],allow_pickle=False) as reference:assert len(reference.files)==32 and all(np.array_equal(reference[f'v{i}'],opt[f'v{i}']) for i in range(32))
  with np.load(folder/'initial-optimizer.npz',allow_pickle=False) as initialopt:
   assert len(initialopt.files)==36 and int(initialopt['v0'])==0 and float(initialopt['v1'])==float(np.float32(1e-5)) and all(np.count_nonzero(initialopt[f'v{i}'])==0 for i in range(2,36))
  schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];identities=[]
  with (folder/'history.jsonl').open() as history,gzip.open(d['auxiliary_schedule']['path'],'rt') as aux:
   for step,(line,aline) in enumerate(zip(history,aux,strict=True),1):
    row=json.loads(line);ar=json.loads(aline);main=schedule[step-1];assert row['step']==row['optimizer_iteration']==step==ar['step'] and row['main_tensor_sha256']==main['phase_sha256'] and row['main_receipt_sha256']==canonical(main['original_receipt']) and row['aux_receipt_sha256']==canonical(ar) and row['aux_input_sha256']==ar['input_sha256'] and len(row['losses'])==2 and np.isfinite(row['losses']).all()
    if step<=2:assert row['losses']==fresh['history'][step-1]['losses']
    identities.append((row['main_tensor_sha256'],row['aux_receipt_sha256']))
   assert step==10000
  if target in matched:assert identities==matched[target]
  else:matched[target]=identities
  checks.append(dict(target=target,condition=condition,all10000_main_and_auxiliary_receipts_exact=True,initial_state_original_and_fresh=True,all10snapshots_audited=True,all36adam_arrays_finite=True,iterations=10000,final_equals_snapshot10000=True,old_control_numeric_exact=condition=='control'))
 out=dict(source_result_sha256=sha(RESULT),execution_sha256=sha(LOCK),auditor_sha256=sha(__file__),all_passed=True,checks=checks,actual_main_batches=40000,actual_auxiliary_batches=40000,limits='Independent stored H5/state/Adam/snapshot/receipt/paired-input audit. Autodiff, gradient connections and runtime order signatures remain producer evidence; NPZ inventory order is not an independently reconstructed H5 path order. No export/INT8/recognition/newdata/promotion claim.')
 with (R/'4090-engine-adversarial-formal-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
