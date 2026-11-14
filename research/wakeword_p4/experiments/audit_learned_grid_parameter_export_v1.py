"""Static original folded parameter compiler roundtrip; zero model invokes."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from compile_learned_grid_parameters_v1 import compile_parameters
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lp=R/'4090-matched-internal-boundaries-execution-v1.json';mp=R/'4090-learned-activation-grid-constraints-v1.json';lock=json.loads(lp.read_text());mapping=json.loads(mp.read_text());rows=[]
 for e in lock['exports']:
  its=[]
  for path in (e['fp32_path'],e['path']):
   assert sha(path)==lock['files'][path];it=Interpreter(model_path=path,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();its.append(it)
  params=[]
  for op in its[0]._get_ops_details():
   if op['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED'):params.append(tuple(its[0].get_tensor(int(i)) for i in op['inputs'][1:]))
  group=next(x['groups'] for x in mapping['models'] if (x['target'],x['condition'])==(e['target'],e['condition']))
  original=Path(e['path']).read_bytes();compiled,receipt=compile_parameters(original,group,{},params)
  assert compiled==original and len(receipt['constants'])==20 and not receipt['parameter_writes']
  rows.append(dict(target=e['target'],condition=e['condition'],original_sha256=sha(e['path']),bytes_exact=True,constants=receipt['constants'],passed=True))
 out=dict(lock_sha256=sha(lp),mapping_sha256=sha(mp),compiler_sha256=sha(Path(__file__).with_name('compile_learned_grid_parameters_v1.py')),auditor_sha256=sha(__file__),cases=rows,all_passed=True,model_invocations=0,optimizer_updates=0,limits='Original folded FP32 W/b compiler roundtrip only. Exact original bytes, no parameter/step updates or forward invocation. Post-update compiler/actual parity remains unproven.')
 (R/'4090-learned-grid-parameter-export-roundtrip-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),all_passed=True)))
if __name__=='__main__':main()
