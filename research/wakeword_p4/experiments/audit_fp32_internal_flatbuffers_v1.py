"""Read original FlatBuffers only; no interpreter, export, model or media inference."""
import hashlib,json
from pathlib import Path
import numpy as np
from tensorflow.lite.python import schema_py_generated as S
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/fp32-internal-software-v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 result=json.loads((ROOT/'result.json').read_text());rows=[];names={v:k for k,v in vars(S.BuiltinOperator).items() if isinstance(v,int)};types={v:k for k,v in vars(S.TensorType).items() if isinstance(v,int)}
 for r in result['cases']:
  p=ROOT/(r['target']+'-'+r['condition'])/'model.tflite';data=p.read_bytes();assert sha(data)==r['model_sha256'];g=S.Model.GetRootAsModel(data,0);graphs=[];zero=[]
  for gi in range(g.SubgraphsLength()):
   sg=g.Subgraphs(gi);tensors=[];ops=[]
   for ti in range(sg.TensorsLength()):
    t=sg.Tensors(ti);b=g.Buffers(t.Buffer());raw=bytes(b.DataAsNumpy()) if b.DataLength() else b'';q=t.Quantization()
    assert t.Type() not in [S.TensorType.INT8,S.TensorType.UINT8,S.TensorType.INT16]
    assert q is None or q.ScaleLength()==0
    tensors.append(dict(index=ti,name=t.Name().decode(),shape=t.ShapeAsNumpy().tolist(),dtype=types[t.Type()],buffer=int(t.Buffer()),bytes=len(raw),sha256=sha(raw),variable=bool(t.IsVariable())))
    if gi>0 and raw and t.Type()==S.TensorType.FLOAT32:
     assert np.all(np.frombuffer(raw,dtype='<f4')==0);zero.append(dict(graph=gi,tensor=ti,shape=t.ShapeAsNumpy().tolist(),sha256=sha(raw)))
   for oi in range(sg.OperatorsLength()):
    op=sg.Operators(oi);name=names[g.OperatorCodes(op.OpcodeIndex()).BuiltinCode()];assert name not in ['CUSTOM','QUANTIZE','DEQUANTIZE'];entry=dict(index=oi,name=name,inputs=op.InputsAsNumpy().tolist(),outputs=op.OutputsAsNumpy().tolist())
    if name=='FULLY_CONNECTED':
     f=S.FullyConnectedOptions();o=op.BuiltinOptions();f.Init(o.Bytes,o.Pos);entry['fused_activation']=int(f.FusedActivationFunction());assert entry['fused_activation']==S.ActivationFunctionType.NONE
    ops.append(entry)
   graphs.append(dict(index=gi,inputs=sg.InputsAsNumpy().tolist(),outputs=sg.OutputsAsNumpy().tolist(),tensors=tensors,operators=ops))
  main=graphs[0];assert len(main['inputs'])==len(main['outputs'])==1
  for index,shape in [(main['inputs'][0],[1,3,40]),(main['outputs'][0],[1,1])]:assert main['tensors'][index]['shape']==shape and main['tensors'][index]['dtype']=='FLOAT32'
  assert len(zero)==6 and sum(o['name']=='READ_VARIABLE' for o in main['operators'])==6 and sum(o['name']=='ASSIGN_VARIABLE' for o in main['operators'])==6
  assert sum(o['name']=='LOGISTIC' for o in main['operators'])==1
  rows.append(dict(target=r['target'],condition=r['condition'],model_sha256=sha(data),graphs=graphs,six_real_zero_initializer_buffers=zero,static_passed=True))
 out=dict(schema='fp32-internal-flatbuffer-audit-v1',source_sha256=sha(Path(__file__).read_bytes()),original_result_sha256=sha((ROOT/'result.json').read_bytes()),models=rows,limits='Static schema/buffer audit only, not runtime execution, state dynamics, original H5 reconstruction proof, full-stream qualification or recognition. Integer shape/index tensors permitted; no quantized tensors.')
 (R/'4090-fp32-internal-flatbuffer-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(models=len(rows),all_static_passed=True)))
if __name__=='__main__':main()
