"""Mechanical changed-grid compiler, no training/converter/inference.
Bias real values here come from ORIGINAL quantized bias; final training must
instead encode its actual trained folded bias. This is only export feasibility.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/learned-grid-compile-witness-v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compile_bytes(data,groups,selected,factor):
 if factor==1:return data,[]
 model=S.ModelT.InitFromObj(S.Model.GetRootAsModel(data,0));old=copy.deepcopy(model)
 allowed=set();changed=[]
 for gi,group in enumerate(groups):
  if gi not in selected:continue
  new=np.float32(np.float64(group['step'])*np.float64(factor[gi] if isinstance(factor,dict) else factor));assert np.isfinite(new) and new>0
  for member in group['members']:
   t=model.subgraphs[member['graph']].tensors[member['tensor']]
   t.quantization.scale=np.array([new],dtype=np.float32)
   # Zero point fixed: initializer byte codes remain real zero.
 for op in model.subgraphs[0].operators:
  name=model.operatorCodes[op.opcodeIndex].builtinCode
  if name not in (S.BuiltinOperator.CONV_2D,S.BuiltinOperator.DEPTHWISE_CONV_2D,S.BuiltinOperator.FULLY_CONNECTED):continue
  g=model.subgraphs[0];og=old.subgraphs[0];a,w,b=[g.tensors[int(x)] for x in op.inputs];ob=og.tensors[int(op.inputs[2])]
  bs=np.float32(a.quantization.scale[0])*w.quantization.scale
  if np.array_equal(bs,b.quantization.scale):continue
  assert b.type==S.TensorType.INT32 and np.all(b.quantization.zeroPoint==0)
  codes=np.frombuffer(old.buffers[ob.buffer].data.tobytes(),dtype='<i4').astype(np.float64)
  real=codes*ob.quantization.scale.astype(np.float64)
  rounded=np.rint(real/bs.astype(np.float64));assert np.all((rounded>=-(1<<31))&(rounded<(1<<31)))
  original_buffer=b.buffer
  refs=[(gi,ti) for gi,g in enumerate(model.subgraphs) for ti,t in enumerate(g.tensors) if t.buffer==b.buffer]
  if len(refs)>1:
   model.buffers.append(copy.deepcopy(model.buffers[b.buffer]));b.buffer=len(model.buffers)-1
  newcodes=rounded.astype('<i4');model.buffers[b.buffer].data=newcodes.view(np.uint8);b.quantization.scale=bs.copy()
  if b.buffer==original_buffer:allowed.add(b.buffer)
  changed.append(dict(tensor=int(op.inputs[2]),buffer=b.buffer,original_buffer=original_buffer,cloned_shared_buffer=b.buffer!=original_buffer,old_scales=ob.quantization.scale.tolist(),new_scales=bs.tolist(),code_sha256=hashlib.sha256(newcodes.tobytes()).hexdigest()))
 for i,(b,ob) in enumerate(zip(model.buffers[:len(old.buffers)],old.buffers,strict=True)):
  if i not in allowed:assert np.array_equal(b.data,ob.data)
 builder=flatbuffers.Builder(0);offset=model.Pack(builder);builder.Finish(offset,file_identifier=b'TFL3');result=bytes(builder.Output())
 parsed=S.ModelT.InitFromObj(S.Model.GetRootAsModel(result,0))
 # Revert only declared scale/bias changes, then compare full object packing.
 reverted=copy.deepcopy(parsed)
 for gi,group in enumerate(groups):
  if gi not in selected:continue
  for member in group['members']:
   g,t=member['graph'],member['tensor'];reverted.subgraphs[g].tensors[t].quantization.scale=old.subgraphs[g].tensors[t].quantization.scale.copy()
 for record in changed:
  t=record['tensor'];reverted.subgraphs[0].tensors[t].quantization.scale=old.subgraphs[0].tensors[t].quantization.scale.copy();i=record['original_buffer'];reverted.subgraphs[0].tensors[t].buffer=i;reverted.buffers[i].data=old.buffers[i].data.copy()
 reverted.buffers=reverted.buffers[:len(old.buffers)]
 def pack(m):
  b=flatbuffers.Builder(0);o=m.Pack(b);b.Finish(o,file_identifier=b'TFL3');return bytes(b.Output())
 assert pack(reverted)==pack(old),'unallowed graph or data mutation'
 return result,changed

