"""Static FP32 compiler no-op roundtrip across all4original models."""
import hashlib,json
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from compile_learned_grid_float_parameters_v1 import compile_parameters
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lp=R/'4090-matched-internal-boundaries-execution-v1.json';p=json.loads(lp.read_text());rows=[]
 for e in p['exports']:
  path=e['fp32_path'];assert sha(path)==p['files'][path];it=Interpreter(model_path=path,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors()
  ops=[o for o in it._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')];values=[tuple(it.get_tensor(int(i)) for i in o['inputs'][1:]) for o in ops]
  original=Path(path).read_bytes();compiled,receipt=compile_parameters(original,values);assert original==compiled and not receipt['writes'] and len(receipt['constants'])==20
  rows.append(dict(target=e['target'],condition=e['condition'],source_sha256=sha(path),original_bytes_exact=True,passed=True))
 out=dict(lock_sha256=sha(lp),compiler_sha256=sha(Path(__file__).with_name('compile_learned_grid_float_parameters_v1.py')),auditor_sha256=sha(__file__),cases=rows,all_passed=True,model_invocations=0,limits='No-op original constant roundtrip only. Posttraining FP32/INT8 complete-waveform numerical qualification still required.')
 (R/'4090-learned-grid-float-roundtrip-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=4,all_passed=True)))
if __name__=='__main__':main()
