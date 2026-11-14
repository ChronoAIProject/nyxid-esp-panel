"""Saved-capture numerical boundary decomposition, never model inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/matched-internal-boundaries-v1');OUT=Path('.local/wakeword/matched-internal-decomposition-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def real(a,m):
 if not m['scales']:return a.astype(np.float64)
 assert len(m['scales'])==len(m['zeros'])==1
 return (a.astype(np.float64)-m['zeros'][0])*m['scales'][0]
def sigmoid(a):return np.exp(-np.logaddexp(0.,-a))
def stats(a):
 a=np.asarray(a).reshape(-1);v=np.abs(np.convolve(a,np.ones(5)/5,'valid'))
 return dict(raw_max=float(np.max(np.abs(a))),five_mean=float(v.mean()),five_p99=float(np.percentile(v,99)),five_max=float(v.max()))
def main():
 p=ROOT/'result.json';r=json.loads(p.read_text());assert r['all_passed'] and len(r['cases'])==4;OUT.mkdir(exist_ok=False);rows=[]
 for c in r['cases']:
  assert sha(c['captures_path'])==c['captures_sha256'] and sha(c['metadata_path'])==c['metadata_sha256'];meta=json.loads(Path(c['metadata_path']).read_text())
  with np.load(c['captures_path'],allow_pickle=False) as a:
   for backend in ['fp32','int8']:
    for key in meta[backend]:
     if key.startswith('head_'):continue
     arr=a[backend+'_'+key];assert arr[0].tobytes()==arr[1].tobytes() and np.isfinite(arr).all()
   layers=[]
   for i in range(9):
    key=f'encoder_{i:02}';f=real(a['fp32_'+key],meta['fp32'][key]);q=real(a['int8_'+key],meta['int8'][key]);assert f.shape==q.shape
    layers.append(dict(layer=i,shape=list(f.shape[2:]),dequant_max_abs=float(np.max(np.abs(f-q))),dequant_mean_abs=float(np.mean(np.abs(f-q)))))
   embedding=real(a['int8_embedding'],meta['int8']['embedding']).reshape(2,-1,int(np.prod(meta['int8']['embedding']['shape'])))
   fw=real(a['fp32_head_weight'],meta['fp32']['head_weight']);fb=real(a['fp32_head_bias'],meta['fp32']['head_bias'])
   qw=real(a['int8_head_weight'],meta['int8']['head_weight']);qb=real(a['int8_head_bias'],meta['int8']['head_bias'])
   A=real(a['fp32_probability'],meta['fp32']['probability']).reshape(2,-1)
   B=sigmoid(embedding@fw.T+fb).reshape(2,-1);C=sigmoid(embedding@qw.T+qb).reshape(2,-1)
   D=sigmoid(real(a['int8_logit'],meta['int8']['logit'])).reshape(2,-1)
   E=real(a['int8_probability'],meta['int8']['probability']).reshape(2,-1)
   boundaries=dict(quantized_embedding_float64_float_head=B-A,head_weight_bias=C-B,integer_fc_and_float_accumulation=D-C,logistic_output=E-D)
   assert np.max(np.abs(sum(boundaries.values())-(E-A)))<1e-12
   for x in [A,B,C,D,E]:assert x[0].tobytes()==x[1].tobytes() and np.isfinite(x).all()
   path=OUT/(c['target']+'-'+c['condition']+'.npz');np.savez(path,A_fp32=A,B_quant_embedding_float_head=B,C_quant_embedding_quant_head=C,D_actual_int8_logit_sigmoid=D,E_actual_uint8=E,**boundaries)
   rows.append(dict(target=c['target'],condition=c['condition'],request_order=0,layers=layers,boundaries={k:stats(v[0]) for k,v in boundaries.items()},total=stats((E-A)[0]),arrays_path=str(path),arrays_sha256=sha(path)))
 result=dict(original_capture_result_sha256=sha(p),cases=rows,model_invocations=0,limits='Four prospective first-request witnesses only. Signed numerical telescope, not causal percentages or all540 attribution. B/C use float64 diagnostic heads; D-C includes floating accumulation differences as well as actual integer FC. Per-layer dequant differences do not independently identify clipping/cause. All original outputs/rejections remain unchanged; no injected activations, training, export, threshold or data changes.')
 (R/'4090-matched-internal-boundaries-original-result-v1.json').write_bytes(p.read_bytes())
 (R/'4090-matched-internal-decomposition-v1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps([{k:c[k] for k in ['target','condition','boundaries','total']} for c in rows]))
if __name__=='__main__':main()
