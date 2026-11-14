"""Independent saved-array FC/gradient/repeat audit; no TensorFlow or model invoke."""
import json,hashlib
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-scalar-fc-shared-network-result-v3.json';result=json.loads(p.read_text());plan=R/'4090-scalar-fc-shared-network-plan-v3.json';assert result['plan_sha256']==sha(plan) and result['all_passed'] and len(result['cases'])==8
 lock=json.loads(plan.read_text())
 for path,h in lock['files'].items():assert sha(path)==h,path
 arms=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json').read_text())['arms'];templates=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json').read_text())['arms'];seen=set();reports=[]
 for arm in arms:
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();details={d['index']:d for d in i.get_tensor_details()};ops=next(t for t in templates if (t['target'],t['condition'])==(arm['target'],arm['condition']))['template']['ops'];fc=next(o for o in ops if o['op']=='FULLY_CONNECTED');xi,wi,bi=map(int,fc['inputs']);ins,iz=details[xi]['quantization'];os,oz=details[65]['quantization'];ws=float(details[wi]['quantization_parameters']['scales'][0]);w=i.get_tensor(wi).astype(np.int64);b=i.get_tensor(bi).astype(np.int64)
  for c in result['cases']:
   if (c['target'],c['condition'])!=(arm['target'],arm['condition']):continue
   key=(c['target'],c['condition'],c['batch']);assert key not in seen;seen.add(key);assert sha(c['arrays_path'])==c['arrays_sha256']
   with np.load(c['arrays_path'],allow_pickle=False) as a:
    keys=[k[3:] for k in a.files if k.startswith('r0_')];assert all(np.array_equal(a['r0_'+k],a['r1_'+k]) and np.isfinite(a['r0_'+k]).all() for k in keys)
    assert np.array_equal(a['original_logits'],a['r0_logits']) and np.array_equal(a['original_probability'],a['r0_probability'])
    for op in (o for o in ops if o['op'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')):
     for ordinal,n in enumerate(op['inputs'][1:]):
      v=a['r0_'+str(n)];sc=details[n]['quantization_parameters']['scales'].astype(float)
      if ordinal==0:
       if op['op']=='DEPTHWISE_CONV_2D':v=np.transpose(v,(3,0,1,2));sc=sc.reshape(1,1,1,-1)
       elif op['op']=='CONV_2D':v=np.transpose(v,(3,0,1,2));sc=sc.reshape(-1,1,1,1)
       else:v=v.T;sc=sc.reshape(-1,1)
      assert np.array_equal(np.rint(v.astype(float)/sc),i.get_tensor(n))
    embedding=np.rint(a['r0_embedding_dequantized'].astype(float)/ins).astype(np.int64);acc=embedding@w.T+b;assert np.array_equal(acc,a['r0_scalar_fc_accumulator']) and np.array_equal(acc,a['independent_accumulator']);real=acc.astype(float)*(ins*ws/os);expected=np.clip(np.sign(real)*np.floor(np.abs(real)+.5)+oz,max(-128,oz),min(127,int(np.rint(6/os))+oz)).astype(np.int64);assert np.array_equal(expected,a['independent_fc_codes']) and np.array_equal(expected,np.rint(a['r0_scalar_fc_exact']/os+oz).astype(np.int64))
    gradients=[a[f'r0_gradient_{n}'] for n in range(15)];norms=[float(np.linalg.norm(g.astype(float))) for g in gradients];assert norms==c['gradient_l2'];reports.append(dict(target=c['target'],condition=c['condition'],batch=c['batch'],constants=20,fc_samples=len(acc),repeat_arrays=len(keys),gradient_tensors=15,zero_gradient_tensors=sum(n==0 for n in norms),all_saved_checks_passed=True))
 assert len(seen)==8
 with (R/'4090-scalar-fc-shared-network-independent-audit-v3.json').open('x') as f:json.dump(dict(result_sha256=sha(p),cases=reports,model_invocations=0,limits='Independent saved hashes/all20constants/integerFC/gradient norms/finite/repeats/raw preservation under NEWpolicy. Actual autodiff and model/BN unchanged remain live producer evidence; no independent network execution or mathematical correctness of wholegradient chain, recognition or admission.'),f,indent=2);f.write('\n')
 print('All8 saved cases audited independently')
if __name__=='__main__':main()
