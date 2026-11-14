"""Direct folded trainable parameter export atop linked activation groups.
No converter/calibration; parameter arrays must follow conv/depthwise/FC order.
"""
import copy,hashlib
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S
from compile_learned_grid_nonuniform_v1 import compile_bytes

def pack(model):
 b=flatbuffers.Builder(0);o=model.Pack(b);b.Finish(o,file_identifier=b'TFL3');return bytes(b.Output())
def compile_parameters(data,groups,factors,parameters):
 base,scale_changes=compile_bytes(data,groups,list(factors),factors) if any(v!=1 for v in factors.values()) else (data,[])
 model=S.ModelT.InitFromObj(S.Model.GetRootAsModel(base,0));before=copy.deepcopy(model);g=model.subgraphs[0]
 ops=[op for op in g.operators if model.operatorCodes[op.opcodeIndex].builtinCode in (S.BuiltinOperator.CONV_2D,S.BuiltinOperator.DEPTHWISE_CONV_2D,S.BuiltinOperator.FULLY_CONNECTED)]
 assert len(ops)==len(parameters)==10
 writes=[];records=[]
 for op,pair in zip(ops,parameters,strict=True):
  for ordinal,(index,value) in enumerate(zip(op.inputs[1:],pair,strict=True)):
   index=int(index);t=g.tensors[index];v=np.asarray(value);q=t.quantization;assert tuple(t.shape)==v.shape and np.isfinite(v).all()
   shape=[1]*v.ndim;shape[q.quantizedDimension]=len(q.scale);sc=q.scale.reshape(shape);zp=q.zeroPoint.reshape(shape)
   bias=ordinal==1;calc=np.float64 if bias else np.float32;low,high=(-(1<<31),(1<<31)-1) if bias else (-127,127)
   raw=np.rint(v.astype(calc)/sc.astype(calc))+zp.astype(calc)
   if bias:assert np.all((raw>=low)&(raw<=high)),'bias overflow'
   codes=np.clip(raw,low,high).astype('<i4' if bias else 'i1');encoded=codes.view(np.uint8).reshape(-1)
   assert len(encoded)==len(model.buffers[t.buffer].data)
   records.append(dict(tensor=index,role='bias' if bias else 'weight',code_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),clipped=int(np.count_nonzero((raw<low)|(raw>high)))))
   if np.array_equal(encoded,model.buffers[t.buffer].data):continue
   original_buffer=t.buffer
   refs=[(gi,ti) for gi,sub in enumerate(model.subgraphs) for ti,x in enumerate(sub.tensors) if x.buffer==original_buffer]
   if len(refs)>1:model.buffers.append(copy.deepcopy(model.buffers[original_buffer]));t.buffer=len(model.buffers)-1
   model.buffers[t.buffer].data=encoded.copy();writes.append(dict(tensor=index,original_buffer=original_buffer,buffer=t.buffer))
 # Verify no undeclared object/data changes beyond explicit parameter writes.
 reverted=copy.deepcopy(model)
 for change in writes:
  i=change['original_buffer'];reverted.subgraphs[0].tensors[change['tensor']].buffer=i;reverted.buffers[i].data=before.buffers[i].data.copy()
 reverted.buffers=reverted.buffers[:len(before.buffers)];assert pack(reverted)==pack(before)
 result=pack(model) if writes else base
 return result,dict(scale_bias_changes=scale_changes,parameter_writes=writes,constants=records,unallowed_mutation_check=True)
