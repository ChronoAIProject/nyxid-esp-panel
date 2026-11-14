"""Interface regression against existing failed-model traces; no new experiment."""
import json
from pathlib import Path
import numpy as np
from collect_engine_adversarial_development_v1 import R,sha,prepare,predict

def main():
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 sources=[R/'4090-pooled-matched-export-original-result-v1.json',R/'4090-pooled-matched-screen-result-v1.json',R/'4090-hey-broad-trained-quantization-plan-v1.json']
 exports,screen,plan=[json.loads(p.read_text()) for p in sources]
 rows=[]
 for target in ['hey_nyx','hey_willow']:
  model=next(m for m in exports['models'] if m['target']==target and m['condition']=='flat')['artifacts']['int8']
  case=next(c for c in screen['cases'] if c['target']==target and c['condition']=='flat' and c['request_order']==0)
  request=next(r for r in plan['requests'] if r['order']==0)
  assert sha(model['path'])==model['sha256'] and sha(case['arrays_path'])==case['arrays_sha256']
  def runtime(**kwargs):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kwargs)
  with np.load(case['arrays_path'],allow_pickle=False) as a:
   values,support,clocks=prepare(a['raw'],request['samples'])
   first,codes,info=predict(values,model['path'],runtime);second,repeat,again=predict(values,model['path'],runtime)
   assert codes.tobytes()==repeat.tobytes()==a['uint8'].tobytes()==a['uint8_repeat'].tobytes()
   assert first.tobytes()==second.tobytes() and info==again
   assert np.array_equal(clocks,a['sample_end']) and np.array_equal(support,a['support_unclamped'])
   decoded=((a['uint8'].astype(np.float64)-case['output_quantization'][1])*case['output_quantization'][0]).astype(np.float32)
   assert decoded.tobytes()==first.tobytes() and info['output']==case['output_quantization'] and info['input']==case['input_quantization']
   assert info['quantized_input_sha256']==__import__('hashlib').sha256(a['quantized_input'].tobytes()).hexdigest()
  rows.append(dict(target=target,model_sha256=model['sha256'],arrays_sha256=case['arrays_sha256'],steps=len(codes),original_failed_case_passed=case['passed'],interface_exact=True))
 report=dict(files={str(p):sha(p) for p in sources+[Path(__file__),Path(__file__).with_name('collect_engine_adversarial_development_v1.py')]},cases=rows,all_interface_checks_passed=True,limits='Two already consumed original flat-model first cases, two fresh interpreters each. Existing failed INT8 numerical/recognition status unchanged. Checks collector runtime/input-code/clock interface only; no engine-adversarial model, training update, new audio, source qualification or cross-platform equality claim.')
 with (R/'4090-engine-int8-collector-interface-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':main()
