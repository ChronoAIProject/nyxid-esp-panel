"""All four learned tensor20 constants: CPU/GPU division and round-even identity."""
import json,hashlib
from pathlib import Path
import numpy as np,h5py
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/scalar-fc-constant-rounding-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import hey_broad_gpu_training_factory_v1 as helper
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;OUT.mkdir(exist_ok=False);reports=[]
 compile=R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json';formal=R/'4090-hey-broad-consistency-formal-result-v1.json';cases=json.loads(formal.read_text())['cases']
 for arm in json.loads(compile.read_text())['arms']:
  c=next(c for c in cases if (c['target'],c['condition'])==(arm['target'],arm['condition']));h5=c['final_repeat']['weights_path'];assert sha(h5)==arm['final_h5_sha256'] and sha(arm['path'])==arm['sha256']
  with h5py.File(h5) as f:raw=f['layers/depthwise_conv2d_3/vars/0'][:];value=np.transpose(raw,(3,0,1,2))
  interpreter=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors();detail=next(d for d in interpreter.get_tensor_details() if d['index']==20);scale=detail['quantization_parameters']['scales'].astype(np.float32).reshape(1,1,1,-1);actual=interpreter.get_tensor(20)
  quotient=value/scale;codes=np.clip(np.rint(quotient),-127,127).astype(np.int8);assert np.array_equal(codes,actual)
  arrays=dict(value=value,scale=scale,numpy_quotient=quotient,compiled_codes=actual);modes={}
  for device in ('/CPU:0','/GPU:0'):
   for mode in ('eager','function'):
    def calculate(v,s):
     q=v/s;return q,tf.clip_by_value(tf.math.rint(q),-127.,127.)
    with tf.device(device):
     v=tf.constant(value);s=tf.constant(scale);q,code=(tf.function(calculate) if mode=='function' else calculate)(v,s)
    name=device[1:4].lower()+'_'+mode;arrays[name+'_quotient']=q.numpy();arrays[name+'_codes']=code.numpy().astype(np.int8);diff=arrays[name+'_codes'].astype(np.int64)-actual.astype(np.int64);indices=np.argwhere(diff!=0)
    modes[name]=dict(unequal=int(len(indices)),maximum_code_difference=int(np.abs(diff).max()),differences=[dict(index=idx.tolist(),numpy_quotient=float(quotient[tuple(idx)]),tensorflow_quotient=float(q.numpy()[tuple(idx)]),compiled_code=int(actual[tuple(idx)]),tensorflow_code=int(code.numpy()[tuple(idx)]),value=float(value[tuple(idx)]),scale=float(scale[0,0,0,idx[-1]])) for idx in indices])
  dest=OUT/f'{arm["target"]}-{arm["condition"]}.npz';np.savez_compressed(dest,**arrays);reports.append(dict(target=arm['target'],condition=arm['condition'],h5_sha256=sha(h5),model_sha256=arm['sha256'],tensor=20,numpy_compiler_exact=True,modes=modes,arrays_path=str(dest),arrays_sha256=sha(dest)));print(arm['target'],arm['condition'],{k:v['unequal'] for k,v in modes.items()},flush=True)
 with (R/'4090-scalar-fc-constant-rounding-diagnostic-v1.json').open('x') as f:json.dump(dict(files={str(p):sha(p) for p in (Path(__file__).resolve().relative_to(Path.cwd()),compile,formal)},cases=reports,model_invocations=0,optimizer_updates=0,limits='Allfour actual learned tensor20 arrays and unchanged model constants. CPU/GPU eager/function scalar-operation reproduction only, not fullnetwork replay, other19constant identities or causal effect on wholewave/recognition. No altered export/recompile/training or tolerance. Firstfailure preserved.'),f,indent=2);f.write('\n')
if __name__=='__main__':main()
