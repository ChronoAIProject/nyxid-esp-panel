"""Temporal depthwise convolutions decomposition using immutable constants; no invoke."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from ai_edge_litert import schema_py_generated as S
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(a,m):
 q=m['quantization_parameters'];s=q['scales'];z=q['zero_points']
 if len(s)==0:return a.astype('float64')
 shape=[1]*a.ndim;shape[int(q['quantized_dimension'])]=len(s)
 return (a.astype('float64')-z.reshape(shape))*s.reshape(shape)
def stats(x):return dict(mean_abs=float(abs(x).mean()),max_abs=float(abs(x).max()))
def main():
 lock=R/'4090-matched-internal-boundaries-execution-v1.json';p=json.loads(lock.read_text());cap=R/'4090-matched-internal-boundaries-original-result-v1.json';c=json.loads(cap.read_text());rows=[]
 for row in c['cases']:
  e=next(x for x in p['exports'] if (x['target'],x['condition'])==(row['target'],row['condition']))
  assert sha(row['captures_path'])==row['captures_sha256'] and sha(row['metadata_path'])==row['metadata_sha256']
  meta=json.loads(Path(row['metadata_path']).read_text());constants={};identities=[]
  for backend,key in [('fp32','fp32_path'),('int8','path')]:
   path=e[key];assert sha(path)==p['files'][path]
   it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors()
   ops=[x for x in it._get_ops_details() if x['op_name']=='DEPTHWISE_CONV_2D'];assert len(ops)==4
   ds={x['index']:x for x in it.get_tensor_details()};constants[backend]=[]
   model=S.Model.GetRootAsModel(Path(path).read_bytes(),0);sg=model.Subgraphs(0)
   schemaops=[sg.Operators(i) for i in range(sg.OperatorsLength()) if model.OperatorCodes(sg.Operators(i).OpcodeIndex()).BuiltinCode()==S.BuiltinOperator.DEPTHWISE_CONV_2D]
   # All initializer constants must encode real zero, not integer zero assumed.
   initializers=0
   for gi in range(1,model.SubgraphsLength()):
    sub=model.Subgraphs(gi)
    for ti in range(sub.TensorsLength()):
     t=sub.Tensors(ti);buf=model.Buffers(t.Buffer())
     if not buf.DataLength() or t.Type() not in (S.TensorType.FLOAT32,S.TensorType.INT8):continue
     raw=np.frombuffer(bytes(buf.DataAsNumpy()),dtype='<f4' if t.Type()==S.TensorType.FLOAT32 else 'int8');q=t.Quantization()
     zero=0 if q is None or not q.ZeroPointLength() else q.ZeroPoint(0)
     assert np.all(raw==zero);initializers+=1
   assert initializers==6
   for layer,op,sop in zip((1,3,5,7),ops,schemaops,strict=True):
    assert int(op['outputs'][0])==meta[backend][f'encoder_{layer:02}']['index']
    options=S.DepthwiseConv2DOptions();o=sop.BuiltinOptions();options.Init(o.Bytes,o.Pos)
    assert options.DilationHFactor()==options.DilationWFactor()==options.StrideH()==options.StrideW()==options.DepthMultiplier()==1
    assert options.Padding()==S.Padding.VALID and options.FusedActivationFunction()==S.ActivationFunctionType.NONE
    cat,=[x for x in it._get_ops_details() if int(op['inputs'][0]) in x['outputs']]
    assert cat['op_name']=='CONCATENATION' and int(cat['inputs'][1])==meta[backend][f'encoder_{layer-1:02}']['index']
    history_shape=ds[int(cat['inputs'][0])]['shape']

    w,b=[decode(it.get_tensor(int(i)),ds[int(i)]) for i in op['inputs'][1:]]
    assert w.shape[0]==w.shape[2]==1 and list(history_shape)==[1,w.shape[1]-1,1,w.shape[3]]
    constants[backend].append((w[0,:,0,:],b))
   identities.append(dict(backend=backend,path=path,sha256=sha(path)))
  with np.load(row['captures_path'],allow_pickle=False) as a:
   def get(backend,k):
    v=a[backend+'_'+k][0].astype('float64');m=meta[backend][k]
    return v if not m['scales'] else (v-m['zeros'][0])*m['scales'][0]
   for j,layer in enumerate((1,3,5,7)):
    wf,bf=constants['fp32'][j];wq,bq=constants['int8'][j]
    xf=get('fp32',f'encoder_{layer-1:02}').reshape(-1,wf.shape[1]);xq=get('int8',f'encoder_{layer-1:02}').reshape(xf.shape)
    yf=get('fp32',f'encoder_{layer:02}').reshape(-1,wf.shape[1]);yq=get('int8',f'encoder_{layer:02}').reshape(yf.shape)
    def local(x,w,b):
     padded=np.pad(x,((len(w)-1,0),(0,0)))
     return np.stack([(padded[t:t+len(w)]*w).sum(axis=0)+b for t in range(len(x))])
    A=local(xf,wf,bf);B=local(xq,wf,bf);C=local(xq,wq,bq)
    reconstruction=stats(A-yf);assert reconstruction['max_abs']<1e-3,reconstruction
    terms={'upstream_with_float_weights':B-A,'weight_bias_quantization':C-B,'integer_output_and_accumulation':yq-C,'float_reconstruction':A-yf}
    assert np.max(abs(sum(terms.values())-(yq-yf)))<1e-10
    rows.append(dict(target=row['target'],condition=row['condition'],layer=layer,models=identities,float_reconstruction=reconstruction,boundaries={k:stats(v) for k,v in terms.items()},total=stats(yq-yf),input_error=stats(xq-xf),float_weight_temporal_abs_sum_max=float(abs(wf).sum(axis=0).max()),float_weight_temporal_abs_sum_median=float(np.median(abs(wf).sum(axis=0)))))
 out=dict(lock_sha256=sha(lock),capture_report_sha256=sha(cap),script_sha256=sha(__file__),model_invocations=0,cases=rows,limits='Four original first requests, full startup. Local float64 causal depthwise decomposition; actual dilation/stride1, VALID, no fused activation, concat history length and six real-zero initializer constants checked; signed telescope not causal percentages. Integer residual includes requantization and accumulation. No changed model/injection/training/calibration/new data or qualification.')
 (R/'4090-matched-depthwise-diagnostic-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([dict(target=r['target'],condition=r['condition'],layer=r['layer'],input_mae=r['input_error']['mean_abs'],upstream_mae=r['boundaries']['upstream_with_float_weights']['mean_abs'],weights_mae=r['boundaries']['weight_bias_quantization']['mean_abs'],residual_mae=r['boundaries']['integer_output_and_accumulation']['mean_abs'],total_mae=r['total']['mean_abs'],weight_gain=r['float_weight_temporal_abs_sum_max']) for r in rows],indent=2))
if __name__=='__main__':main()
