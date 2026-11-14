"""Read-only schema constraints for learned steps, including cross-subgraph state."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as S
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=R/'4090-matched-internal-boundaries-execution-v1.json';p=json.loads(lock.read_text());rows=[]
 for e in p['exports']:
  path=e['path'];assert sha(path)==p['files'][path]
  m=S.Model.GetRootAsModel(Path(path).read_bytes(),0);names={v:k for k,v in vars(S.BuiltinOperator).items() if isinstance(v,int)}
  tensors={};ops=[];handles={};parent={};edges=[]
  for gi in range(m.SubgraphsLength()):
   g=m.Subgraphs(gi)
   for ti in range(g.TensorsLength()):
    t=g.Tensors(ti);q=t.Quantization();key=(gi,ti)
    if q is not None and q.ScaleLength() and t.Type() in (S.TensorType.INT8,S.TensorType.UINT8):
     tensors[key]=dict(graph=gi,tensor=ti,name=t.Name().decode(),scales=q.ScaleAsNumpy().tolist(),zeros=q.ZeroPointAsNumpy().tolist(),constant=bool(m.Buffers(t.Buffer()).DataLength()))
     parent[key]=key
   for oi in range(g.OperatorsLength()):
    o=g.Operators(oi);name=names[m.OperatorCodes(o.OpcodeIndex()).BuiltinCode()];ins=[(gi,int(x)) for x in o.InputsAsNumpy()];outs=[(gi,int(x)) for x in o.OutputsAsNumpy()];ops.append((gi,oi,name,ins,outs))
    if name=='VAR_HANDLE':
     v=S.VarHandleOptions();b=o.BuiltinOptions();v.Init(b.Bytes,b.Pos)
     handles[outs[0]]=(v.Container(),v.SharedName())
  def root(x):
   while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
   return x
  def join(keys,reason):
   assert keys and all(x in tensors for x in keys),(reason,keys)
   for x in keys[1:]:parent[root(x)]=root(keys[0])
   edges.append(dict(reason=reason,tensors=[list(x) for x in keys]))
  states={}
  for gi,oi,name,ins,outs in ops:
   reason=f'{gi}:{oi}:{name}'
   if name=='CONCATENATION':join(ins+outs,reason)
   if name in ('RESHAPE','STRIDED_SLICE','AVERAGE_POOL_2D','EXPAND_DIMS','SQUEEZE','TRANSPOSE','MAX_POOL_2D') and ins[0] in tensors:join([ins[0],outs[0]],reason)
   if name in ('READ_VARIABLE','ASSIGN_VARIABLE'):
    handle=handles[ins[0]];value=outs[0] if name=='READ_VARIABLE' else ins[1]
    assert value in tensors
    states.setdefault(handle,[]).append(value)
  for (container,name),keys in states.items():join(keys,'shared state '+str((container,name)))
  groups={}
  for key in tensors:groups.setdefault(root(key),[]).append(key)
  # Constants used as operator kernels are not activation groups.
  kernelkeys={ins[1] for _,_,name,ins,_ in ops if name in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')}
  biasrows=[]
  for gi,oi,name,ins,outs in ops:
   if name not in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED'):continue
   g=m.Subgraphs(gi);a=g.Tensors(ins[0][1]).Quantization();w=g.Tensors(ins[1][1]).Quantization();b=g.Tensors(ins[2][1]).Quantization()
   product=np.float32(a.Scale(0))*w.ScaleAsNumpy()
   np.testing.assert_allclose(b.ScaleAsNumpy(),product,rtol=2e-7,atol=0)
   biasrows.append(dict(op=oi,kind=name,input_tensor=list(ins[0]),weight_tensor=list(ins[1]),bias_tensor=list(ins[2]),bias_scale_tracks_input_step=True))
  resultgroups=[]
  for keys in groups.values():
   if any(k in kernelkeys for k in keys):continue
   base=tensors[keys[0]]
   assert len(base['scales'])==len(base['zeros'])==1
   for k in keys:assert tensors[k]['scales']==base['scales'] and tensors[k]['zeros']==base['zeros'],keys
   resultgroups.append(dict(step=base['scales'][0],zero=base['zeros'][0],members=[tensors[k] for k in keys]))
  assert len(states)==6 and len(biasrows)==10
  rows.append(dict(target=e['target'],condition=e['condition'],model_sha256=sha(path),groups=resultgroups,equality_constraints=edges,bias_constraints=biasrows,state_variables=len(states),passed=True))
 out=dict(lock_sha256=sha(lock),script_sha256=sha(__file__),models=rows,all_passed=True,model_invocations=0,limits='Read-only schema equality and bias-product audit, not changed-grid compiler, training-graph derivative or actual export arithmetic parity. Kernel scales excluded from learned activation groups. Logistic/input/output freeze policy still requires explicit assignment before training.')
 (R/'4090-learned-activation-grid-constraints-v1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps([dict(target=r['target'],condition=r['condition'],groups=len(r['groups']),states=r['state_variables'],bias_constraints=len(r['bias_constraints'])) for r in rows]))
if __name__=='__main__':main()
