"""Saved-array gradient/sign/state audit; no TensorFlow/autodiff or model imports."""
import hashlib,json,math
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 lock=R/'4090-engine-adversarial-gradient-execution-v1.json';result=R/'4090-engine-adversarial-gradient-result-v1.json';d=json.loads(lock.read_text());r=json.loads(result.read_text());assert r['all_passed'] and not r['optimizer_created'] and r['execution_sha256']==sha(lock) and len(r['targets'])==2
 aux=json.loads(Path(d['input_result']).read_text())['batches'];summaries=[];domain_reference={}
 for target in r['targets']:
  assert target['optimizer_updates']==0 and target['all_state_unchanged'] and len(target['cases'])==50
  assert sha(target['initialization_path'])==target['initialization_sha256'];init=json.loads(Path(target['initialization_path']).read_text());assert init['model_initial_source']==d['initial'] and sha(init['initial_h5_path'])==init['initial_h5_sha256']
  assert sha(init['state_path'])==init['state_sha256'] and sha(target['end_state_path'])==target['end_state_sha256'];inventory=init['inventory'];ti=init['trainable_inventory'];wake=init['wake_indices'];enc=init['encoder_indices'];assert len(ti)==17 and len(wake)==2 and len(enc)==13 and sorted(enc+wake)==list(range(15)) and init['domain_indices']==[15,16]
  with np.load(init['state_path'],allow_pickle=False) as a,np.load(target['end_state_path'],allow_pickle=False) as b:
   assert set(a.files)==set(b.files)=={f'v{i}' for i in range(len(inventory))}
   for rec in inventory:
    key=f"v{rec['index']}";x=a[key];assert list(x.shape)==rec['shape'] and str(x.dtype)==rec['dtype'] and hashlib.sha256(x.tobytes()).hexdigest()==rec['sha256'] and np.isfinite(x).all() and np.array_equal(x,b[key])
   for rec in ti:assert any((rec['name'],rec['shape'],rec['sha256'])==(v['name'],v['shape'],v['sha256']) for v in inventory)
  schedule=[json.loads(l) for l in Path(d['schedules'][target['target']]['path']).read_text().splitlines()];rows=[]
  for i,case in enumerate(target['cases']):
   assert case['step']==i+1 and case['repeat_exact'] and case['original_receipt']==schedule[i]['original_receipt'] and case['main_tensor_sha256']==schedule[i]['phase_sha256'] and case['aux_arrays_sha256']==aux[i]['arrays_sha256']
   assert sha(case['arrays_path'])==case['arrays_sha256'] and sha(aux[i]['arrays_path'])==case['aux_arrays_sha256'];assert case['none_indices']==[[15,16],wake,wake,wake]
   with np.load(case['arrays_path'],allow_pickle=False) as a,np.load(aux[i]['arrays_path'],allow_pickle=False) as ax:
    assert a['main_x'].shape==(128,225,40) and a['main_y'].shape==(128,1) and a['main_weights'].shape==(128,) and a['main_logits'].shape==(128,1) and a['domain_logits'].shape==(33,3)
    assert sum(a['main_y'][:,0]==1)==32 and sum(a['main_y'][:,0]==0)==96 and np.array_equal(a['main_weights'],np.where(a['main_y'][:,0]==1,1,20))
    h=hashlib.sha256()
    for k in ['main_x','main_y','main_weights']:assert a[k].dtype==np.float32;h.update(a[k].tobytes())
    assert h.hexdigest()==case['main_tensor_sha256'] and np.array_equal(a['engine_labels'],ax['engine'])
    z=a['main_logits'][:,0].astype(float);y=a['main_y'][:,0];w=a['main_weights'];S=float(np.mean((np.logaddexp(0,z)-y*z)*w));dz=a['domain_logits'].astype(float);mx=dz.max(axis=1);D=float(np.mean(mx+np.log(np.exp(dz-mx[:,None]).sum(axis=1))-dz[np.arange(33),a['engine_labels']]))
    assert np.allclose(a['losses'],[S,D,D,D],rtol=1e-5,atol=1e-5) and np.array_equal(a['losses'],np.array(case['losses'],np.float32))
    expected={'main_x','main_y','main_weights','main_logits','domain_logits','engine_labels','losses'}
    for tag,none in zip(['S','D','R0','R1'],case['none_indices']):
     for j in range(17):
      key=f'{tag}_{j}'
      if j in none:assert key not in a.files;continue
      expected.add(key);assert list(a[key].shape)==ti[j]['shape'] and str(a[key].dtype)==ti[j]['dtype']
    assert set(a.files)==expected and all(np.isfinite(a[k]).all() for k in a.files)
    for j in enc:assert np.count_nonzero(a[f'R0_{j}'])==0 and np.array_equal(a[f'R1_{j}'],-a[f'D_{j}'])
    for j in [15,16]:assert np.array_equal(a[f'R0_{j}'],a[f'D_{j}']) and np.array_equal(a[f'R1_{j}'],a[f'D_{j}'])
    sg=np.concatenate([a[f'S_{j}'].astype(float).ravel() for j in enc]);dg=np.concatenate([a[f'D_{j}'].astype(float).ravel() for j in enc]);sn=float(np.linalg.norm(sg));dn=float(np.linalg.norm(dg));cos=float(np.dot(sg,dg)/sn/dn) if sn and dn else None
    identity=hashlib.sha256(dz.tobytes()+dg.tobytes()+b''.join(a[f'D_{j}'].tobytes() for j in [15,16])).hexdigest()
    if i in domain_reference:assert domain_reference[i]==identity
    else:domain_reference[i]=identity
    rows.append(dict(step=i+1,supervised_loss=S,engine_loss=D,encoder_supervised_norm=sn,encoder_engine_norm=dn,cosine_S_D=cos,cosine_S_reversed_D=-cos if cos is not None else None))
  sn=float(np.median([x['encoder_supervised_norm'] for x in rows]));dn=float(np.median([x['encoder_engine_norm'] for x in rows]));summaries.append(dict(target=target['target'],batches=50,encoder_supervised_median_norm=sn,encoder_engine_median_norm=dn,median_norm_ratio=sn/dn if dn else None,per_batch=rows))
 out=dict(execution_sha256=sha(lock),original_result_sha256=sha(result),auditor_sha256=sha(__file__),all_passed=True,summaries=summaries,all100_main_batches_authenticated=True,all_state_arrays_unchanged=True,domain_gradients_cross_target_exact=True,coefficient_selected=None,limits='Independent saved-array sign/loss/norm/None-pattern/mainbatch/auxlabel/state-inventory reduction only. Original runtime repeat/autodiff and original-H5-to-runtime loading are producer evidence, not independently rerun network/autodiff or H5 path-order reconstruction. No optimizer/newdata/threshold/efficacy/INT8 qualification.')
 with (R/'4090-engine-adversarial-gradient-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k!='summaries'},indent=2))
 for s in summaries:print(json.dumps({k:v for k,v in s.items() if k!='per_batch'}))
if __name__=='__main__':main()
