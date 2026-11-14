"""No-TF saved-context, repeat, code/layout and reference comparison audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as schema
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pp=R/'4090-bounded-training-forward-witness-plan-v1.json';rp=R/'4090-bounded-training-forward-witness-result-v1.json';p=json.loads(pp.read_text());r=json.loads(rp.read_text());assert r['plan_sha256']==sha(pp) and len(r['cases'])==len(p['cases'])==4;reports=[]
 for path,h in p['files'].items():assert sha(path)==h,path
 for c,row in zip(p['cases'],r['cases'],strict=True):
  assert (c['target'],c['condition'],c['request_order'])==(row['target'],row['condition'],row['request_order']);assert sha(row['arrays_path'])==row['arrays_sha256']
  model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(Path(c['arm']['path']).read_bytes(),0));graph=model.subgraphs[0];summary={}
  with np.load(c['screen']['arrays_path']) as z:raw=z['raw'];q=z['quantized_input'];floating=z['float'][74:];integer=z['uint8'][74:];phase=z['offline_phase'];nominal=z['offline_nominal']
  values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);ends=np.arange(225,len(values)+1,3)
  with np.load(c['reference']['arrays_path']) as ref,np.load(row['arrays_path']) as a:
   assert np.array_equal(a['ends'],ends) and np.array_equal(a['actual_embedding'],ref['64'][74:].reshape(-1,1800)) and np.array_equal(a['actual_fc'],ref['actual_65'][74:].reshape(-1,1))
   for k,v in [('whole_float',floating),('whole_uint8',integer),('offline_phase',phase),('offline_nominal',nominal)]:assert np.array_equal(a[k],v)
   assert all(np.isfinite(a[k]).all() for k in a.files)
   for mode in ['nominal','phase']:
    x=np.stack([values[e-225:e] if mode=='nominal' else np.concatenate([values[e-224:e],np.zeros((1,40),np.float32)]) for e in ends]);assert np.array_equal(a[mode+'_x'],x)
    keys=[k for k in a.files if k.startswith(mode+'_r0_')];assert keys
    for k in keys:assert np.array_equal(a[k],a[k.replace('_r0_','_r1_',1)])
    ops=[o for o in graph.operators if len(o.inputs)==3 and int(o.inputs[1]) in [34,32,30,28,26,24,22,20,18,16]];assert len(ops)==10
    for op in ops:
     for bias,n in enumerate(op.inputs[1:]):
      n=int(n);t=graph.tensors[n];v=a[f'{mode}_r0_{n}'];sc=np.asarray(t.quantization.scale,float)
      if not bias:
       if n in [32,28,24,20]:v=np.transpose(v,(3,0,1,2));sc=sc.reshape(1,1,1,-1)
       elif n==16:v=v.T;sc=sc.reshape(-1,1)
       else:v=np.transpose(v,(3,0,1,2));sc=sc.reshape(-1,1,1,1)
      codes=np.rint(v.astype(float)/sc);actual=np.frombuffer(bytes(model.buffers[t.buffer].data),dtype='<i4' if bias else 'i1').reshape(t.shape);assert np.array_equal(codes,actual)
    egrid=graph.tensors[64].quantization;fgrid=graph.tensors[65].quantization
    ec=np.rint(a[mode+'_r0_embedding_dequantized'].astype(float)/egrid.scale[0]+egrid.zeroPoint[0]).astype(np.int64);fc=np.rint(a[mode+'_r0_scalar_fc_exact'].astype(float)/fgrid.scale[0]+fgrid.zeroPoint[0]).astype(np.int64);assert np.array_equal(ec,a[mode+'_embedding_codes']) and np.array_equal(fc,a[mode+'_fc_codes'])
    summary[mode]=dict(contexts=len(x),embedding_unequal=int(np.sum(ec!=a['actual_embedding'])),embedding_maximum=int(np.max(np.abs(ec-a['actual_embedding']))),fc_unequal=int(np.sum(fc!=a['actual_fc'])),fc_maximum=int(np.max(np.abs(fc-a['actual_fc']))),saved_repeats_exact=True);assert summary[mode]==row['metrics'][mode]
  reports.append(dict(target=c['target'],condition=c['condition'],metrics=summary))
 with (R/'4090-bounded-training-forward-witness-independent-audit-v1.json').open('x') as f:f.write(json.dumps(dict(result_sha256=sha(rp),cases=reports,model_invocations=0,limits='Authenticates saved arrays/contexts/repeats/constants/code conversion and differences to original actual traces. Model-unchanged is producer live evidence only; no independent execution, training, numerical gate or recognition admission.'),indent=2)+'\n')
 print(json.dumps(reports))
if __name__=='__main__':main()
