"""Controlled captured-constant scale vs runtime scale division; no model inference."""
import json,hashlib
from pathlib import Path
import numpy as np,h5py
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/constant-scale-capture-folding-off-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_fake_quant_primitives_v1 import affine_ste
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;OUT.mkdir(exist_ok=False);reports=[]
 compiled=R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json';formal=R/'4090-hey-broad-consistency-formal-result-v1.json';cases=json.loads(formal.read_text())['cases']
 for arm in json.loads(compiled.read_text())['arms']:
  c=next(c for c in cases if (c['target'],c['condition'])==(arm['target'],arm['condition']));h5=c['final_repeat']['weights_path'];assert sha(h5)==arm['final_h5_sha256']
  with h5py.File(h5) as f:value=f['layers/depthwise_conv2d_3/vars/0'][:]
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();detail=next(d for d in i.get_tensor_details() if d['index']==20);scale=detail['quantization_parameters']['scales'].reshape(1,1,-1,1);actual=np.transpose(i.get_tensor(20),(1,2,3,0));arrays=dict(value=value,scale=scale,compiled=actual);modes={}
  for optimize in (False,):
   tf.config.optimizer.set_experimental_options({'arithmetic_optimization':False,'constant_folding':False})
   for device in ('/CPU:0','/GPU:0'):
    with tf.device(device):
     variable=tf.Variable(value)
     @tf.function
     def captured():
      result=affine_ste(variable,scale,np.zeros_like(scale),-127,127,tf)
      return result,variable/tf.constant(scale)
     dequant,quotient=captured()
    name=device[1:4].lower()+'_constant_folding_and_arithmetic_off';code=np.rint(dequant.numpy().astype(np.float64)/scale.astype(np.float64)).astype(np.int64);diff=code-actual;indices=np.argwhere(diff!=0);arrays[name+'_code']=code;arrays[name+'_quotient']=quotient.numpy()
    modes[name]=dict(unequal=len(indices),differences=[dict(index=idx.tolist(),code=int(code[tuple(idx)]),compiled=int(actual[tuple(idx)]),quotient=float(quotient.numpy()[tuple(idx)])) for idx in indices])
  dest=OUT/f'{arm["target"]}-{arm["condition"]}.npz';np.savez_compressed(dest,**arrays);reports.append(dict(target=arm['target'],condition=arm['condition'],modes=modes,arrays_path=str(dest),arrays_sha256=sha(dest)));print(arm['target'],arm['condition'],{k:v['unequal'] for k,v in modes.items()},flush=True)
 with (R/'4090-constant-scale-capture-folding-off-v1.json').open('x') as f:json.dump(dict(files={str(p):sha(p) for p in (Path(__file__).resolve().relative_to(Path.cwd()),Path(__file__).with_name('hey_broad_fake_quant_primitives_v1.py'),compiled,formal)},cases=reports,model_invocations=0,optimizer_updates=0,limits='Controlled isolated actualweight variable+capturedscale fakequant and arithmetic optimizer process-local ablation. Not fullnetwork rerun, no modification to original execution policy/exports, no causal wholewave/recognition claim.'),f,indent=2);f.write('\n')
if __name__=='__main__':main()
