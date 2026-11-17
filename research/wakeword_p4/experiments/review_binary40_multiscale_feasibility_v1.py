"""Independent all-output integer replay of the immutable untrained model."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from causal_source_balanced_batch_v1 import Sampler,digest
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/binary40-multiscale-feasibility-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 report=R/'4090-binary40-multiscale-feasibility-original-v1.json';d=read(report);plan=R/'4090-binary40-multiscale-feasibility-plan-v1.json';assert d['plan_sha256']==sha(plan)
 for path,h in read(plan)['files'].items():assert sha(path)==h,path
 for name,h in d['files'].items():assert sha(ROOT/name)==h,name
 assert d['trained'] is False and d['optimizer_updates']==0 and d['all_neural_tensors_integer'];path=d['model']['path'];assert sha(path)==d['model']['sha256']
 (features,p,n),descriptors=Sampler(726).batch();assert d['calibration_descriptors']==descriptors[:16] and d['original_batch_hashes']==dict(features=digest(features),positive=digest(p),negative=digest(n))
 def fresh():
  it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();return it,it.get_signature_runner()
 it,runner=fresh();inputs=runner.get_input_details();outputs=runner.get_output_details();shapes=[[1,h,1,c] for h,c in zip([2,4,8,16,32,40,1],[40,64,64,64,64,64,384])];states={};tables={};pairs=[]
 for i,shape in enumerate(shapes):
  name=f'cache{i}';a=inputs[name];b=outputs[name+'_out'];assert list(a['shape'])==list(b['shape'])==shape and a['dtype']==b['dtype']==np.int8;s,z=b['quantization'];ts,tz=a['quantization'];assert ts>0 and s>0;states[name]=np.full(shape,tz,np.int8);codes=np.arange(-128,128,dtype=np.float64);tables[name]=np.clip(np.rint((codes-z)*s/ts)+tz,-128,127).astype(np.int8);pairs.append(dict(name=name,input_quantization=list(a['quantization']),output_quantization=list(b['quantization']),bytes=int(np.prod(shape))))
 assert sum(p['bytes'] for p in pairs)==6864
 raw=read(plan);cal=np.load(ROOT/'calibration.npz',allow_pickle=False);assert len(cal.files)==128*8
 for j in range(128):
  for i,shape in enumerate(shapes):assert list(cal[f'r{j}_cache{i}'].shape)==shape
  assert cal[f'r{j}_chunk'].shape==(1,3,1,40)
 expected=np.load(ROOT/'feedback.npz',allow_pickle=False);assert len(expected.files)==256*8;hasher=hashlib.sha256();first=None
 for call in range(256):
  x=features[:1,3*call:3*call+3,None,:];scale,zero=inputs['chunk']['quantization'];q=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8);actual=runner(chunk=q,**states)
  for name,value in sorted(actual.items()):assert np.array_equal(value,expected[f'{call}_{name}']),(call,name);hasher.update(value.tobytes())
  if call==0:first=(q.copy(),{k:v.copy() for k,v in actual.items()})
  states={f'cache{i}':tables[f'cache{i}'][actual[f'cache{i}_out'].astype(np.int16)+128] for i in range(7)}
 assert hasher.hexdigest()==d['feedback_sha256'];other,run2=fresh();again=run2(chunk=first[0],**{f'cache{i}':np.full(shape,inputs[f'cache{i}']['quantization'][1],np.int8) for i,shape in enumerate(shapes)});assert all(np.array_equal(v,again[k]) for k,v in first[1].items());assert sha(path)==d['model']['sha256']
 out=dict(schema='binary40-multiscale-feasibility-review-v1',all_passed=True,original_report_sha256=sha(report),script_sha256=sha(__file__),all256calls_all8outputs_exact=True,lookup_table_cache_feedback_independent=True,cache_pairs=pairs,cache_bytes=6864,all128representative_shapes_checked=True,fresh_reset_all_outputs_exact=True,limits='Independent original-batch/trace replay on untrainedbuiltinINT8 graph only; no recognition, optimizer, trainedquantization, P4arena/latency or human claim.')
 with (R/'4090-binary40-multiscale-feasibility-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
