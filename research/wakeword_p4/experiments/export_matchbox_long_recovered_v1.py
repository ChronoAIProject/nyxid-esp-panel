"""One builtinINT8 export from the fixed final5000 PyTorch checkpoint."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
R=Path('research/wakeword_p4/results');TRAIN=None;OUT=None;LOCK=None;ORIGINAL=None;ARM=None
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pack():
 import torch
 r=json.loads(ORIGINAL.read_text());assert sha(TRAIN/'final.pt')==r['files']['final.pt']['sha256'] and r['all_training_checks_passed'] and r['steps']==5000
 state=torch.load(TRAIN/'final.pt',map_location='cpu',weights_only=True)
 with (TRAIN/'final-for-export.npz').open('xb') as f:np.savez(f,**{n:v.detach().numpy() for n,v in state.items()})
 receipt=dict(source_sha256=sha(TRAIN/'final.pt'),packed_sha256=sha(TRAIN/'final-for-export.npz'),tensors=len(state),model_calls=0,neural_comparison_performed=False)
 with (TRAIN/'export-pack.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
def plan():
 if not ORIGINAL.exists():print(json.dumps(dict(status='not_ready',model_calls=0)));return 2
 r=json.loads(ORIGINAL.read_text());parent=R/'4090-matchbox-long-training-plan-v1.json';p=json.loads(parent.read_text())
 assert r['plan_sha256']==sha(parent) and r['all_training_checks_passed'] and r['steps']==r['optimizer_iterations']==5000
 for name in ['final.pt','frontend.pt','representative.npy']:assert sha(TRAIN/name)==r['files'][name]['sha256']
 rep=np.load(TRAIN/'representative.npy',allow_pickle=False);assert rep.shape==(129,280,1,64) and rep.dtype==np.float32 and np.isfinite(rep).all()
 if not (TRAIN/'export-pack.json').exists():subprocess.run([sys.executable,str(Path(__file__)),'pack','--arm',ARM],check=True)
 packed=json.loads((TRAIN/'export-pack.json').read_text());assert packed['source_sha256']==sha(TRAIN/'final.pt') and packed['packed_sha256']==sha(TRAIN/'final-for-export.npz')
 files=dict(p['files']);paths=[Path(__file__),R/'4090-matchbox-long-control-resume-plan-v1.json',R/'4090-matchbox-long-recovery-plan-v1.json',R/'4090-matchbox-long-original-interruption-v1.json',parent,ORIGINAL,TRAIN/'final.pt',TRAIN/'frontend.pt',TRAIN/'representative.npy',TRAIN/'final-for-export.npz',TRAIN/'export-pack.json']
 files.update({str(x):sha(x) for x in paths})
 lock=dict(schema='matchbox-long-int8-plan-v1',arm=ARM,files=files,versions={n:metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},expected_new_gib=.2,representative_shape=list(rep.shape),retries=0,policy='Fixed final5000only, one builtinINT8 conversion. Exact trained PyTorch arrays unpacked without model inference into bound NPZ; CPU tensor deserialization is not CPU training. Standard symmetric depthwise/pointwise/residual/BN/ReLU/globalmean/7softmax/Nyxslice Keras graph; stored129actual preupdate representative windows, no calibration retry. One fresh interpreter functional call and basic static integer/interface checks only, no FP32 or neuralCPU/GPU numerical comparison gate. Float32MFCC raw-window frontend remains external. Actual complete waveform INT8 diagnostics still required; no freshdev/acceptance/threshold selection/promotion, v30 retained.')
 with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),representative=129,weights_sha256=sha(TRAIN/'final.pt'))),flush=True)
 return 0
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.2'],check=True);OUT.mkdir(exist_ok=False)
 try:
  import tensorflow as tf
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  arrays=np.load(TRAIN/'final-for-export.npz',allow_pickle=False);layers=tf.keras.layers
  inputs=layers.Input(batch_shape=(1,280,1,64),dtype=tf.float32,name='mfcc');x=inputs
  def conv(x,filters,kernel,dilation,weight,name,depthwise=False):
   if depthwise:
    layer=layers.DepthwiseConv2D((kernel,1),padding='same',dilation_rate=(dilation,1),use_bias=False,name=name)
    value=arrays[weight].transpose(2,1,0)[:,:,:,None]
   else:
    layer=layers.Conv2D(filters,(kernel,1),padding='same',dilation_rate=(dilation,1),use_bias=False,name=name)
    value=arrays[weight].transpose(2,1,0)[:,None,:,:]
   y=layer(x);layer.set_weights([value]);return y
  def bn(x,prefix,name):
   layer=layers.BatchNormalization(epsilon=.001,momentum=.9,trainable=False,name=name);y=layer(x,training=False)
   layer.set_weights([arrays[prefix+s] for s in ['.weight','.bias','.running_mean','.running_var']]);return y
  specs=[(128,11,1,True,False),(64,13,1,True,True),(64,15,1,True,True),(64,17,1,True,True),(128,29,2,True,False),(128,1,1,False,False)]
  for i,(filters,kernel,dilation,separable,residual) in enumerate(specs):
   before=x;prefix=f'encoder.encoder.{i}'
   if separable:
    x=conv(x,None,kernel,dilation,prefix+'.mconv.0.conv.weight',f'b{i}_dw',True)
    x=conv(x,filters,1,1,prefix+'.mconv.1.conv.weight',f'b{i}_pw');bnindex=2
   else:x=conv(x,filters,kernel,dilation,prefix+'.mconv.0.conv.weight',f'b{i}_conv');bnindex=1
   x=bn(x,prefix+f'.mconv.{bnindex}',f'b{i}_bn')
   if residual:
    branch=conv(before,filters,1,1,prefix+'.res.0.0.conv.weight',f'b{i}_resconv');branch=bn(branch,prefix+'.res.0.1',f'b{i}_resbn');x=layers.Add(name=f'b{i}_add')([x,branch])
   x=layers.ReLU(name=f'b{i}_relu')(x)
  x=layers.GlobalAveragePooling2D(name='whole_window_mean')(x);head=layers.Dense(7,name='phrase_logits');x=head(x)
  head.set_weights([arrays['decoder.decoder_layers.0.weight'].T,arrays['decoder.decoder_layers.0.bias']]);x=layers.Softmax(name='phrase_softmax')(x)
  output=layers.Lambda(lambda y:y[:,1:2],name='nyx')(x);model=tf.keras.Model(inputs,output)
  rep=np.load(TRAIN/'representative.npy',allow_pickle=False)
  converter=tf.lite.TFLiteConverter.from_keras_model(model);converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.representative_dataset=lambda:([rep[i:i+1]] for i in range(len(rep)))
  converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.uint8
  raw=converter.convert();path=OUT/'int8.tflite';path.write_bytes(raw)
  it=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
  assert a['dtype']==np.int8 and b['dtype']==np.uint8 and list(a['shape'])==[1,280,1,64] and list(b['shape'])==[1,1]
  scale,zero=a['quantization'];assert scale>0 and b['quantization']==(1/256,0)
  details=it.get_tensor_details();assert all(d['dtype'] in (np.int8,np.uint8,np.int32,np.int64) for d in details)
  ops=[x['op_name'] for x in it._get_ops_details()];assert 'SOFTMAX' in ops and 'DEPTHWISE_CONV_2D' in ops and 'CONV_2D' in ops and not any(o.startswith('Flex') for o in ops)
  q=np.clip(np.rint(rep[:1]/scale)+zero,-128,127).astype(np.int8);it.set_tensor(a['index'],q);it.invoke();code=int(it.get_tensor(b['index'])[0,0]);assert 0<=code<=255
  assert sha(TRAIN/'final.pt')==p['files'][str(TRAIN/'final.pt')]
  result=dict(schema='matchbox-long-int8-original-v1',arm=ARM,plan_sha256=sha(LOCK),model=dict(path=str(path),bytes=len(raw),sha256=sha(path)),input_shape=[1,280,1,64],output_shape=[1,1],input_quantization=[scale,zero],output_quantization=list(b['quantization']),ops=ops,all_static_and_runtime_abi_passed=True,neural_model_stateless=True,external_raw_context_samples=44800,representative_windows=129,functional_calls=1,functional_output_code=code,source_weights_file_unchanged=True,model_numerical_comparison_performed=False,recognition_scored=False,acceptance_accessed=False,promotion_qualified=False,policy=p['policy'])
  with (R/f'4090-matchbox-long-{ARM}-int8-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)
 except BaseException:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','pack','run']);parser.add_argument('--arm',choices=['control','pooled'],required=True);a=parser.parse_args()
 ARM=a.arm;TRAIN=Path('.local/wakeword')/f"matchbox-long-{ARM}-{'resumed' if ARM=='control' else 'recovery'}-v1";OUT=Path('.local/wakeword')/f'matchbox-long-{ARM}-int8-v1';LOCK=R/f'4090-matchbox-long-{ARM}-int8-plan-v1.json';ORIGINAL=R/f'4090-matchbox-long-{ARM}-training-recovery-v1.json'
 raise SystemExit({'plan':plan,'pack':pack,'run':run}[a.action]())
