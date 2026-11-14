"""Direct FP32 folded W/b export for honest float vs INT8 qualification.
Raw sigmoid and original graph retained; no converter, calibration or inference.
"""
import copy,hashlib
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S

def pack(model):
 b=flatbuffers.Builder(0);o=model.Pack(b);b.Finish(o,file_identifier=b'TFL3');return bytes(b.Output())
def compile_parameters(data,parameters):
 graph=S.ModelT.InitFromObj(S.Model.GetRootAsModel(data,0));old=copy.deepcopy(graph);g=graph.subgraphs[0]
 ops=[o for o in g.operators if graph.operatorCodes[o.opcodeIndex].builtinCode in (S.BuiltinOperator.CONV_2D,S.BuiltinOperator.DEPTHWISE_CONV_2D,S.BuiltinOperator.FULLY_CONNECTED)];assert len(ops)==len(parameters)==10
 writes=[];constants=[]
 for op,pair in zip(ops,parameters,strict=True):
  for index,value in zip(op.inputs[1:],pair,strict=True):
   index=int(index);t=g.tensors[index];v=np.asarray(value,dtype='<f4');assert t.type==S.TensorType.FLOAT32 and tuple(t.shape)==v.shape and np.isfinite(v).all()
   assert t.quantization is None or t.quantization.scale is None or len(t.quantization.scale)==0
   raw=v.view(np.uint8).reshape(-1);assert len(raw)==len(graph.buffers[t.buffer].data)
   constants.append(dict(tensor=index,sha256=hashlib.sha256(v.tobytes()).hexdigest()))
   if np.array_equal(raw,graph.buffers[t.buffer].data):continue
   original=t.buffer;refs=[(gi,ti) for gi,sub in enumerate(graph.subgraphs) for ti,x in enumerate(sub.tensors) if x.buffer==original]
   if len(refs)>1:graph.buffers.append(copy.deepcopy(graph.buffers[original]));t.buffer=len(graph.buffers)-1
   graph.buffers[t.buffer].data=raw.copy();writes.append(dict(tensor=index,original_buffer=original,buffer=t.buffer))
 reverted=copy.deepcopy(graph)
 for w in writes:
  reverted.subgraphs[0].tensors[w['tensor']].buffer=w['original_buffer'];i=w['original_buffer'];reverted.buffers[i].data=old.buffers[i].data.copy()
 reverted.buffers=reverted.buffers[:len(old.buffers)];assert pack(reverted)==pack(old)
 return (pack(graph) if writes else data),dict(constants=constants,writes=writes,unallowed_mutation_check=True,limits='float32 serialization of actual folded parameters, original raw sigmoid graph. No learned activation quantizers in FP32; learned steps affect only INT8 route.')
