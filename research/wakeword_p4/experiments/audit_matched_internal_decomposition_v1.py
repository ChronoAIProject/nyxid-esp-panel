"""Independent saved-only audit; never imports reducer or invokes a model."""
import hashlib,json,math
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(a,m):
 if not m['scales']:return a.astype('float64')
 assert len(m['scales'])==len(m['zeros'])==1
 return (a.astype('float64')-m['zeros'][0])*m['scales'][0]
def sig(x):return 1/(1+np.exp(-x))
def stats(x):
 x=x.flatten();five=np.abs(np.array([math.fsum(x[i:i+5])/5 for i in range(len(x)-4)]))
 return dict(raw_max=float(abs(x).max()),five_mean=float(five.mean()),five_p99=float(np.percentile(five,99)),five_max=float(five.max()))
def main():
 cap=R/'4090-matched-internal-boundaries-original-result-v1.json';dec=R/'4090-matched-internal-decomposition-v1.json';lock=R/'4090-matched-internal-boundaries-execution-v1.json'
 c=json.loads(cap.read_text());d=json.loads(dec.read_text());p=json.loads(lock.read_text())
 assert c['execution_sha256']==sha(lock) and d['original_capture_result_sha256']==sha(cap)
 rows=[]
 for source,row in zip(c['cases'],d['cases'],strict=True):
  identity=(source['target'],source['condition'],0)
  assert identity==(row['target'],row['condition'],row['request_order'])
  original=next(x for x in p['cases'] if (x['target'],x['condition'],x['request_order'])==identity)
  for path,key in [('captures_path','captures_sha256'),('metadata_path','metadata_sha256')]:assert sha(source[path])==source[key]
  assert sha(original['arrays_path'])==source['source_arrays_sha256']==original['arrays_sha256']
  assert sha(row['arrays_path'])==row['arrays_sha256']
  m=json.loads(Path(source['metadata_path']).read_text())
  with np.load(source['captures_path'],allow_pickle=False) as a,np.load(original['arrays_path'],allow_pickle=False) as old,np.load(row['arrays_path'],allow_pickle=False) as derived:
   for backend in ('fp32','int8'):
    for key in m[backend]:
     v=a[backend+'_'+key];assert np.isfinite(v).all()
     if not key.startswith('head_'):assert v[0].tobytes()==v[1].tobytes()
    assert a[backend+'_probability'][0].tobytes()==old['fp32' if backend=='fp32' else 'uint8'].tobytes()
   def get(b,k):return decode(a[b+'_'+k],m[b][k])
   emb=get('int8','embedding').reshape(2,-1,np.prod(m['int8']['embedding']['shape']))
   heads=[]
   for b in ('fp32','int8'):
    w=get(b,'head_weight').reshape(-1);bias=get(b,'head_bias').item()
    heads.append(sig(np.sum(emb*w,axis=-1)+bias))
   A=get('fp32','probability').reshape(2,-1);B,C=heads;D=sig(get('int8','logit').reshape(2,-1));E=get('int8','probability').reshape(2,-1)
   for key,val in zip(('A_fp32','B_quant_embedding_float_head','C_quant_embedding_quant_head','D_actual_int8_logit_sigmoid','E_actual_uint8'),(A,B,C,D,E)):
    np.testing.assert_allclose(derived[key],val,rtol=0,atol=2e-14)
   for key,val in zip(row['boundaries'],(B-A,C-B,D-C,E-D)):
    np.testing.assert_allclose(derived[key],val,rtol=0,atol=2e-14)
    for metric,value in stats(val[0]).items():assert abs(value-row['boundaries'][key][metric])<2e-14
   for metric,value in stats((E-A)[0]).items():assert abs(value-row['total'][metric])<2e-14
   for i,layer in enumerate(row['layers']):
    error=abs(get('fp32',f'encoder_{i:02}')-get('int8',f'encoder_{i:02}'))
    assert float(error.max())==layer['dequant_max_abs'] and float(error.mean())==layer['dequant_mean_abs']
   rows.append(dict(target=identity[0],condition=identity[1],passed=True))
 assert len(rows)==4
 out=dict(capture_report_sha256=sha(cap),decomposition_sha256=sha(dec),auditor_sha256=sha(__file__),cases=rows,all_passed=True,model_invocations=0,limits='Independent saved-array/head arithmetic and rolling fsum verification. Four fixed first requests only; no causal attribution or candidate qualification.')
 (R/'4090-matched-internal-decomposition-independent-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
