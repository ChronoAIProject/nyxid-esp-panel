"""Whole fixed Willow14 surrogate/actual operator residual, no optimization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results'); P=R/'4090-hey-broad-surrogate-residual-plan-v1.json'; O=Path('.local/wakeword/hey-broad-surrogate-residual-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-quantized-stream-plan-v1.json'; base=json.loads(parent.read_text());files=dict(base['files'])
 for p in [parent,Path(__file__),Path(__file__).with_name('hey_broad_quantized_stream_trace_v1.py'),R/'4090-hey-broad-quantized-stream-result-v1.json']:files[str(p)]=sha(p)
 P.write_text(json.dumps(dict(files=files,arm=base['arms'][0],export=base['export'],actual='.local/wakeword/hey-broad-equalized-screen-v1/hey_willow-014.npz',surrogate='.local/wakeword/hey-broad-quantized-stream-v1/hey_willow-014.npz',policy='Only unique failed full270 surrogate witness Willow request14; complete saved input twice fresh model/interpreter zero states, same builtinref. Diagnostic copy adds observations only. Require exact prior surrogate probability/logit and actual UINT8 before interpreting per-layer code differences. Save all activation and six state tensors every step. No optimizer/export/newdata/tolerance changes. Layer differences descriptive, not additive causal attribution.'),indent=2)+'\n')
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True)
 assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip()
 lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import probe_hey_context_transfer_shapes_v1 as helper
 from hey_broad_quantized_stream_trace_v1 import configuration,operators
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf; exports=json.loads(Path(lock['export']).read_text())['arms'];arm=next(a for a in exports if a['target']=='hey_willow');cfg=configuration(arm)
 with np.load(lock['actual']) as a:raw=a['raw'];q=a['quantized_input'];expected=a['uint8']
 with np.load(lock['surrogate']) as a:expectedp=a['probability'];expectedl=a['logit']
 values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
 shapes=[[1,2,1,40],[1,4,1,30],[1,8,1,60],[1,12,1,60],[1,20,1,60],[1,29,1,60]]; runs=[];grids={}
 O.mkdir(exist_ok=False)
 for repeat in range(2):
  model=helper.create(225);model.load_weights(lock['arm']['path'])
  for l in model.layers:
   c=getattr(l,'cell',l)
   if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
  before=[v.numpy().copy() for v in model.variables]
  f=tf.function(operators(model,cfg,tf,True),input_signature=[tf.TensorSpec([1,3,40],tf.float32),[tf.TensorSpec(s,tf.float32) for s in shapes]])
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);i.allocate_tensors();details={d['index']:d for d in i.get_tensor_details()};inp=i.get_input_details()[0]['index'];out=i.get_output_details()[0]['index']
  states=[tf.zeros(s,tf.float32) for s in shapes];saved={};scores=[];probs=[];logits=[]
  for at in range(0,len(values),3):
   l,p,states,obs=f(values[None,at:at+3],states);probs.append(float(p.numpy()[0,0]));logits.append(float(l.numpy()[0,0]));i.set_tensor(inp,q[None,at:at+3]);i.invoke();scores.append(int(i.get_tensor(out)[0,0]))
   for k,v in obs.items():
    n=int(k);s,z=details[n]['quantization'];assert s>0;grids[k]=dict(scale=s,zero=z,name=details[n]['name'])
    saved.setdefault('surrogate_'+k,[]).append(v.numpy());saved.setdefault('actual_'+k,[]).append(i.get_tensor(n).copy())
   for j,state in enumerate(states):saved.setdefault('state_'+str(j),[]).append(state.numpy())
  assert np.array_equal(probs,expectedp) and np.array_equal(logits,expectedl) and np.array_equal(scores,expected)
  assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
  runs.append({k:np.stack(v) for k,v in saved.items()});print('repeat',repeat,'exact prior outputs',flush=True)
 assert all(np.array_equal(runs[0][k],runs[1][k]) for k in runs[0])
 path=O/'hey_willow-014.npz';np.savez_compressed(path,**runs[0]);layers=[]
 for k,g in grids.items():
  s=runs[0]['surrogate_'+k];a=runs[0]['actual_'+k];codes=np.rint(s/g['scale']+g['zero']);err=np.abs(codes-a.astype(np.float64));flat=err.reshape(len(err),-1)
  layers.append(dict(index=int(k),grid=g,maximum_code_error=float(err.max()),unequal_fraction=float(np.mean(err!=0)),per_step_maximum=flat.max(axis=1).tolist(),per_step_mean=flat.mean(axis=1).tolist()))
 (R/'4090-hey-broad-surrogate-residual-result-v1.json').write_text(json.dumps(dict(plan_sha256=sha(P),arrays_path=str(path),arrays_sha256=sha(path),prior_outputs_exact=True,all_activation_state_repeats_exact=True,variables_unchanged=True,layers=layers,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);v=a.parse_args();plan() if v.action=='plan' else run()
