"""Cross-check FlatBuffer census against the host interpreter metadata; no invokes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results');p=R/'4090-c161-binary40-static-abi-original-v1.json';d=json.loads(p.read_text());checks={}
for name,model in d['models'].items():
 raw=Path(model['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==model['sha256']
 it=Interpreter(model_path=model['path'],num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors()
 details={x['index']:x for x in it.get_tensor_details()};count=0
 for t in model['subgraphs'][0]['tensors']:
  x=details[t['index']];assert list(x['shape'])==t['shape']
  if t['logical_bytes'] is not None:assert np.dtype(x['dtype']).itemsize*np.prod(x['shape'])==t['logical_bytes'];count+=1
 if name=='binary40':
  sig=it.get_signature_runner();ins,outs=sig.get_input_details(),sig.get_output_details()
  for pair in d['binary40_cache_pairs']:
   a=ins[pair['name']];b=outs[pair['name']+'_out'];assert a['quantization']==b['quantization']==(pair['scale'],pair['real_zero_code'])
   assert a['index']==pair['input_tensor'] and b['index']==pair['output_tensor']
 checks[name]=dict(allocate_tensors_passed=True,logical_tensor_sizes_checked=count,invocations=0)
out=dict(schema='c161-binary40-static-abi-review-v1',all_passed=True,report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checks=checks,limits='Host BUILTIN_REF1thread allocation/metadata only; no model invoke. This does not prove P4 kernels, arena, speed, numerical output or full-panel fit.')
with (R/'4090-c161-binary40-static-abi-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
