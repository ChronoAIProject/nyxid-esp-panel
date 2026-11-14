"""Original FP32 folded constants -> original INT8 codes, static only."""
import hashlib,json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lp=R/'4090-matched-internal-boundaries-execution-v1.json';lock=json.loads(lp.read_text());rows=[]
 for e in lock['exports']:
  its=[]
  for path in (e['fp32_path'],e['path']):
   assert sha(path)==lock['files'][path]
   it=Interpreter(model_path=path,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();its.append(it)
  ops=[[o for o in it._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')] for it in its];assert len(ops[0])==len(ops[1])==10
  ds={d['index']:d for d in its[1].get_tensor_details()};constants=[]
  for order,(f,q) in enumerate(zip(*ops,strict=True)):
   assert f['op_name']==q['op_name']
   for ordinal,(fi,qi) in enumerate(zip(f['inputs'][1:],q['inputs'][1:],strict=True)):
    fi=int(fi);qi=int(qi);v=its[0].get_tensor(fi);actual=its[1].get_tensor(qi);meta=ds[qi]['quantization_parameters'];shape=[1]*v.ndim;shape[meta['quantized_dimension']]=len(meta['scales']);sc=meta['scales'].reshape(shape);z=meta['zero_points'].reshape(shape)
    low,high=(-(1<<31),(1<<31)-1) if ordinal else (-127,127)
    calc=np.float64 if ordinal else np.float32
    scaled=v.astype(calc)/sc.astype(calc);rounded=np.rint(scaled)+z.astype(calc);codes=np.clip(rounded,low,high).astype(actual.dtype)
    constants.append(dict(operator=order,kind=f['op_name'],role='bias' if ordinal else 'weight',fp32_tensor=fi,int8_tensor=qi,elements=v.size,mismatch=int(np.count_nonzero(codes!=actual)),max_code_error=int(np.max(abs(codes.astype(np.int64)-actual.astype(np.int64)))),source_sha256=hashlib.sha256(v.tobytes()).hexdigest(),clipped=int(np.count_nonzero((rounded<low)|(rounded>high)))))
  rows.append(dict(target=e['target'],condition=e['condition'],fp32_sha256=sha(e['fp32_path']),int8_sha256=sha(e['path']),constants=constants,all_codes_exact=all(x['mismatch']==0 for x in constants)))
 out=dict(lock_sha256=sha(lp),auditor_sha256=sha(__file__),cases=rows,all_codes_exact=all(x['all_codes_exact'] for x in rows),model_invocations=0,optimizer_updates=0,limits='Static original FP32 exported folded parameter encoding audit, not complete H5/BN variable reconstruction or trained forward/export parity. Failures preserved without adjusting weights/grids.')
 (R/'4090-learned-grid-parameter-initialization-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([dict(target=x['target'],condition=x['condition'],all_codes_exact=x['all_codes_exact'],nonexact=[c for c in x['constants'] if c['mismatch']]) for x in rows]))
if __name__=='__main__':main()
