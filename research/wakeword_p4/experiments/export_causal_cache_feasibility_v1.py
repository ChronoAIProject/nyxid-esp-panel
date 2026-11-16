"""Untrained builtinINT8 cache ABI feasibility; not a wakeword model."""
import argparse,hashlib,importlib.metadata as metadata,json,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/causal-cache-int8-feasibility-v2');LOCK=R/'4090-causal-cache-int8-feasibility-plan-v2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 files=[Path(__file__),Path('research/wakeword_p4/training/causal_cached_encoder_v1.py'),R/'4090-causal-cache-clock-function-v1.json']
 p=dict(files={str(f):sha(f) for f in files},versions={n:metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},seed=726,representatives=64,feedback_calls=256,policy='Untrained TensorFlowgraph with six causaldepthwise5tap/pointwise64residualReLUblocks and explicitpastcaches4*d,d1/2/4/8/16/32. RandomKerasinitializationseed726, no transfer ofPyTorchweights/recognitionclaim. Analyticuniform[-1,1]features/caches for64calibrationexamples, not realactivationcalibration. SinglebuiltinINT8conversion; allinput/outputcachequantparams anddtypes audited. Feedbackrequantizationexplicit ifscalesdiffer; zeroreset usesinputzeropoints.256analyticcontinuouscalls plusfreshinterpreterresetrepeat. No neuralCPU/GPU/fullchunk equalitygate, training/audio/heldoutdata/thresholdselection orqualification. ExternalMFCCfloat andP4runtime/arena remainunproven.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(sha(LOCK))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True);B.mkdir(exist_ok=False)
 import tensorflow as tf
 from ai_edge_litert.interpreter import Interpreter
 tf.keras.utils.set_random_seed(p['seed']);L=tf.keras.layers;chunk=L.Input(batch_shape=(1,3,1,64),name='chunk');x=chunk;inputs=[chunk];outputs={};histories=[4,8,16,32,64,128]
 for i,h in enumerate(histories):
  state=L.Input(batch_shape=(1,h,1,64),name=f'cache{i}');inputs.append(state);joined=L.Concatenate(axis=1)([state,x]);outputs[f'cache{i}_out']=L.Lambda(lambda z,h=h:z[:,-h:,:,:],name=f'cache{i}_out')(joined)
  y=L.DepthwiseConv2D((5,1),dilation_rate=(h//4,1),padding='valid',use_bias=True,name=f'dw{i}')(joined);y=L.ReLU()(y);y=L.Conv2D(64,1,use_bias=True,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([y,x]))
 outputs['encoded']=x;model=tf.keras.Model(inputs,outputs)
 def representative():
  rng=np.random.default_rng(p['seed'])
  for _ in range(p['representatives']):yield {v.name.split(':')[0]:rng.uniform(-1,1,size=tuple(v.shape)).astype(np.float32) for v in inputs}
 c=tf.lite.TFLiteConverter.from_keras_model(model);c.optimizations=[tf.lite.Optimize.DEFAULT];c.representative_dataset=representative;c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
 raw=c.convert();path=B/'prototype.tflite';path.write_bytes(raw)
 def interpreter():
  it=Interpreter(model_path=str(path),num_threads=1);it.allocate_tensors();runner=it.get_signature_runner();return it,runner
 it,runner=interpreter();a=runner.get_input_details();b=runner.get_output_details();assert set(a)=={'chunk',*[f'cache{i}' for i in range(6)]};assert set(b)=={'encoded',*[f'cache{i}_out' for i in range(6)]}
 assert all(d['dtype'] in [np.int8,np.int32,np.int64] for d in it.get_tensor_details());ops=[d['op_name'] for d in it._get_ops_details()];assert not any(o.startswith('Flex') for o in ops)
 def detail(d):return dict(shape=list(map(int,d['shape'])),dtype=str(d['dtype']),scale=float(d['quantization'][0]),zero=int(d['quantization'][1]))
 for d in [*a.values(),*b.values()]:assert d['dtype']==np.int8 and d['quantization'][0]>0
 state={f'cache{i}':np.full(a[f'cache{i}']['shape'],a[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};rng=np.random.default_rng(1726);digest=hashlib.sha256();first=None;requant=[]
 for i in range(6):
  ai=a[f'cache{i}'];bo=b[f'cache{i}_out'];assert list(ai['shape'])==list(bo['shape'])==[1,histories[i],1,64];requant.append(ai['quantization']!=bo['quantization'])
 for step in range(p['feedback_calls']):
  scale,zero=a['chunk']['quantization'];values=rng.uniform(-1,1,size=(1,3,1,64));q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);result=runner(chunk=q,**state)
  if step==0:first=(q.copy(),{k:v.copy() for k,v in result.items()})
  for name,v in sorted(result.items()):assert v.dtype==np.int8;digest.update(v.tobytes())
  for i in range(6):
   oscale,ozero=b[f'cache{i}_out']['quantization'];iscale,izero=a[f'cache{i}']['quantization'];v=result[f'cache{i}_out'];state[f'cache{i}']=np.clip(np.rint((v.astype(np.float64)-ozero)*oscale/iscale)+izero,-128,127).astype(np.int8)
 # Repeat only same runtime reset semantics, never crossdevice/model equivalence.
 fresh,rr=interpreter();zeros={f'cache{i}':np.full(a[f'cache{i}']['shape'],a[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};again=rr(chunk=first[0],**zeros);assert all(np.array_equal(v,again[k]) for k,v in first[1].items())
 out=dict(schema='causal-cache-int8-feasibility-original-v1',plan_sha256=sha(LOCK),model=dict(path=str(path),bytes=len(raw),sha256=sha(path)),inputs={k:detail(v) for k,v in a.items()},outputs={k:detail(v) for k,v in b.items()},ops=ops,cache_feedback_requantization_required=requant,cache_input_bytes=sum(int(np.prod(a[f'cache{i}']['shape'])) for i in range(6)),analytic_feedback_calls=p['feedback_calls'],feedback_output_sha256=digest.hexdigest(),fresh_same_runtime_reset_passed=True,all_neural_tensors_integer=True,trained=False,recognition_calls=0,promotion_qualified=False,policy=p['policy'])
 with (R/'4090-causal-cache-int8-feasibility-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
