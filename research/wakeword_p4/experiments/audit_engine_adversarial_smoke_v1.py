"""Independent stored-state/H5/gradient-merge and repeat audit; no TensorFlow."""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h5(p):
 out={}
 with h5py.File(p) as f:
  def visit(k,v):
   if isinstance(v,h5py.Dataset):out[k]=v[()]
  f.visititems(visit)
 return out
def inventory(a,records):
 assert set(a.files)=={f'v{i}' for i in range(len(records))}
 for rec in records:
  v=a[f"v{rec['index']}"];assert list(v.shape)==rec['shape'] and str(v.dtype)==rec['dtype'] and hashlib.sha256(v.tobytes()).hexdigest()==rec['sha256'] and np.isfinite(v).all()
def main():
 result=R/'4090-engine-adversarial-smoke-result-v1.json';lock=R/'4090-engine-adversarial-smoke-execution-v1.json';r=json.loads(result.read_text());d=json.loads(lock.read_text());assert r['all_passed'] and r['execution_sha256']==sha(lock) and len(r['cases'])==4
 original=h5(d['initial']['path']);assert sha(d['initial']['path'])==d['initial']['sha256'];probe=json.loads(Path(d['probe_result']).read_text());old=json.loads(Path(d['old_smoke']).read_text());aux=json.loads(Path(d['input_result']).read_text())['batches'];checks=[];matched={};finals={}
 for case in r['cases']:
  target=case['target'];condition=case['condition'];coefficient=d['coefficient'][condition];assert case['coefficient']==coefficient and len(case['repeats'])==2
  first=next(t for t in probe['targets'] if t['target']==target)['cases'][0];assert sha(first['arrays_path'])==first['arrays_sha256'];baseline=next(c for c in old['cases'] if c['target']==target and c['condition']=='full')['repeats'][0];schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()]
  values=[]
  for rep in case['repeats']:
   folder=Path(rep['folder']);assert rep['optimizer_iterations']==2 and rep['bn_unchanged'] and len(rep['history'])==2
   for name,digest in rep['files'].items():assert sha(folder/name)==digest
   initial=h5(folder/'initial.weights.h5');final=h5(folder/'final.weights.h5');assert initial.keys()==final.keys()==original.keys()
   assert all(np.array_equal(initial[k],original[k]) for k in original)
   assert all(np.isfinite(v).all() for v in final.values())
   bn=[k for k in original if 'batch_normalization' in k];assert len(bn)==16 and all(np.array_equal(final[k],original[k]) for k in bn)
   if condition=='control':
    assert sha(baseline['weights_path'])==baseline['weights_sha256'];reference=h5(baseline['weights_path']);assert all(np.array_equal(final[k],reference[k]) for k in final)
   with np.load(folder/'initial-state.npz',allow_pickle=False) as a,np.load(folder/'final-state.npz',allow_pickle=False) as b,np.load(folder/'optimizer.npz',allow_pickle=False) as opt:
    inventory(a,rep['initial_inventory']);inventory(b,rep['final_inventory']);inventory(opt,rep['optimizer_inventory']);assert len(opt.files)==36 and int(opt['v0'])==2 and float(opt['v1'])==float(np.float32(1e-5));assert not np.array_equal(a[f'v{len(a.files)-2}'],b[f'v{len(b.files)-2}'])
    assert rep['optimizer_signature']==hashlib.sha256(b''.join(opt[f'v{i}'].tobytes() for i in range(36))).hexdigest()
    assert rep['state_signature']==hashlib.sha256(b''.join(b[f'v{i}'].tobytes() for i in range(len(b.files)))).hexdigest()
    if condition=='control':assert hashlib.sha256(b''.join(opt[f'v{i}'].tobytes() for i in range(32))).hexdigest()==baseline['optimizer_signature']
    values.append(({k:b[k].copy() for k in b.files},{k:opt[k].copy() for k in opt.files}))
   enc=rep['encoder_indices'];wake=rep['wake_indices'];assert len(enc)==13 and len(wake)==2 and sorted(enc+wake)==list(range(15))
   for i,row in enumerate(rep['history']):
    assert row['step']==i+1 and row['receipt']==schedule[i]['original_receipt'] and row['main_tensor_sha256']==schedule[i]['phase_sha256'] and row['aux_arrays_sha256']==aux[i]['arrays_sha256'];assert sha(row['arrays_path'])==row['arrays_sha256']
    with np.load(row['arrays_path'],allow_pickle=False) as a,np.load(aux[i]['arrays_path'],allow_pickle=False) as auxiliary:
     assert a['main_x'].shape==(128,225,40) and a['main_y'].shape==(128,1) and a['main_weights'].shape==(128,);h=hashlib.sha256()
     for k in ['main_x','main_y','main_weights']:h.update(a[k].tobytes())
     assert h.hexdigest()==row['main_tensor_sha256'] and np.array_equal(a['engine_labels'],auxiliary['engine']);assert np.array_equal(a['losses'],np.array(row['losses'],np.float32))
     expected={'main_x','main_y','main_weights','main_logits','domain_logits','engine_labels','losses'}
     for j in range(17):
      for tag,none in [('S',[15,16]),('D',wake),('applied',[])]:
       key=f'{tag}_{j}'
       if j in none:assert key not in a.files
       else:
        expected.add(key);assert list(a[key].shape)==rep['trainable_inventory'][j]['shape'] and str(a[key].dtype)==rep['trainable_inventory'][j]['dtype']
      want=a[f'D_{j}'] if j>=15 else a[f'S_{j}'] if j in wake or coefficient==0 else a[f'S_{j}']-a[f'D_{j}'];assert np.array_equal(a[f'applied_{j}'],want)
     assert set(a.files)==expected and all(np.isfinite(a[k]).all() for k in a.files)
     if i==0:
      with np.load(first['arrays_path'],allow_pickle=False) as p:
       assert all(np.array_equal(a[k],p[k]) for k in p.files if k.startswith(('S_','D_')) or k in ['main_x','main_y','main_weights','main_logits','domain_logits','engine_labels']);assert np.array_equal(a['losses'],p['losses'][:2])
     identity=(row['main_tensor_sha256'],row['aux_arrays_sha256']);key=(target,i)
     if key in matched:assert matched[key]==identity
     else:matched[key]=identity
   checks.append(dict(target=target,condition=condition,repeat=rep['repeat'],all_saved_components_merged_exact=True,initial_original_h5_exact=True,frozen_bn_arrays=16,optimizer_arrays=36,iterations=2))
  assert all(np.array_equal(values[0][j][k],values[1][j][k]) for j in [0,1] for k in values[0][j]);finals[target,condition]=values[0][0]
 for target in ['hey_nyx','hey_willow']:assert any(not np.array_equal(finals[target,'control'][k],finals[target,'adversarial'][k]) for k in finals[target,'control'])
 out=dict(source_result_sha256=sha(result),execution_sha256=sha(lock),auditor_sha256=sha(__file__),all_passed=True,checks=checks,actual_optimizer_updates=16,all_control_h5_equal_original_full_smoke=True,all_repeat_state_and_adam_arrays_exact=True,limits='Independent stored-gradient merge/state/H5/Adam/firstcase-probe arithmetic audit. Autodiff and per-update runtime trajectories are producer evidence; no independently reconstructed Adam transition, full-run repeat, export/INT8/recognition/freshdata qualification.')
 with (R/'4090-engine-adversarial-smoke-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
