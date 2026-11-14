"""Saved embedding counterfactual head decomposition; float64 diagnostic arithmetic."""
import json,hashlib
from pathlib import Path
import h5py,numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probability(u):return np.exp(-np.logaddexp(0,-((np.clip(u,0,6)-3)*(8/3))))
def stats(delta):
 v=np.abs(np.convolve(delta,np.ones(5)/5,'valid'))
 return dict(mean=float(v.mean()),p99=float(np.quantile(v,.99)),maximum=float(v.max()))
def main():
 recursive=R/'4090-hey-broad-fixed-grid-recursive-result-v1.json';screen=R/'4090-hey-broad-fixed-grid-screen-result-v1.json';plan=R/'4090-hey-broad-fixed-grid-compile-plan-v1.json';meta=R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json'
 rows=json.loads(recursive.read_text())['cases'];original={(c['target'],c['condition'],c['request_order']):c for c in json.loads(screen.read_text())['cases']};arms=json.loads(plan.read_text())['arms'];models=json.loads(meta.read_text())['arms'];cases=[]
 assert len(rows)==540
 for arm,m in zip(arms,models,strict=True):
  assert (arm['target'],arm['condition'])==(m['target'],m['condition']);assert sha(m['path'])==m['sha256'] and sha(arm['weights'])==m['final_h5_sha256']
  i=Interpreter(model_path=m['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();details={d['index']:d for d in i.get_tensor_details()}
  fc,=[op for op in arm['template']['ops'] if op['op']=='FULLY_CONNECTED'];ix,wi,bi=fc['inputs'];oy=fc['outputs'][0]
  ins,inz=details[ix]['quantization'];outs,outz=details[oy]['quantization'];wscale=details[wi]['quantization_parameters']['scales'].astype(np.float64);bscale=details[bi]['quantization_parameters']['scales'].astype(np.float64)
  w=i.get_tensor(wi).astype(np.float64);b=i.get_tensor(bi).astype(np.float64)
  assert np.all(details[wi]['quantization_parameters']['zero_points']==0) and np.all(details[bi]['quantization_parameters']['zero_points']==0)
  with h5py.File(arm['weights']) as f:
   fw=(f['layers/dense/vars/0'][()]*np.float32(.375)).astype(np.float64);fb=(f['layers/dense/vars/1'][()]*np.float32(.375)+np.float32(3)).astype(np.float64)
  for row in (c for c in rows if (c['target'],c['condition'])==(arm['target'],arm['condition'])):
   key=(row['target'],row['condition'],row['request_order']);o=original[key];assert sha(row['arrays_path'])==row['arrays_sha256'] and sha(o['arrays_path'])==o['arrays_sha256']
   with np.load(row['arrays_path'],allow_pickle=False) as a,np.load(o['arrays_path'],allow_pickle=False) as saved:
    embedding=(a['64'].reshape(-1,1800).astype(np.float64)-inz)*ins
    floathead=probability((embedding@fw+fb).reshape(-1))
    quanthead=probability((embedding@(w*wscale[:,None]).T+b*bscale).reshape(-1))
    actualfc=probability((a['actual_65'].reshape(-1).astype(np.float64)-outz)*outs)
    os,oz=o['output_quantization'];actual=(saved['uint8'].astype(np.float64)-oz)*os;floating=saved['float']
    assert np.array_equal(a['actual_uint8'],saved['uint8'])
    pieces=dict(quant_embedding_float_head_minus_original=floathead-floating,weight_bias_quantization=quanthead-floathead,fc_requantization=actualfc-quanthead,downstream=actual-actualfc,total=actual-floating)
    assert np.allclose(sum(v for k,v in pieces.items() if k!='total'),pieces['total'],rtol=0,atol=1e-14)
    cases.append(dict(target=row['target'],condition=row['condition'],request_order=row['request_order'],original_passed=o['passed'],five_error={k:stats(v) for k,v in pieces.items()},float_head_to_quant_head_probability_sign_flips=int(np.sum((floathead>=.5)!=(quanthead>=.5)))))
 dest=R/'4090-hey-broad-fixed-grid-head-decomposition-v1.json'
 with dest.open('x') as f:json.dump(dict(recursive_sha256=sha(recursive),screen_sha256=sha(screen),cases=cases,limits='Counterfactual float64 head arithmetic on saved quantized embedding. Float32 transformed W,b authenticated from finalH5; actual compiled W/b and grids read only. No model invocation or exact TF float-kernel equivalence claim. Nonlinear decomposition is descriptive, not independent causal contributions. Original gates unchanged; no newdev or admission.'),f,indent=2);f.write('\n')
 print(str(dest),len(cases))
if __name__=='__main__':main()
