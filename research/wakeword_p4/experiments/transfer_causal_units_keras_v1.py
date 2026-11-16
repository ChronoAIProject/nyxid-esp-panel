"""Exact trained weight transfer and CUDA PyTorch/CPU Keras logit comparison."""
import hashlib,json,os,sys
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel

def build(state,classes=34):
 L=tf.keras.layers;chunk=L.Input(batch_shape=(1,3,1,64),name='chunk');x=chunk;inputs=[chunk];outputs={}
 for i,h in enumerate([4,8,16,32,64,128]):
  cache=L.Input(batch_shape=(1,h,1,64),name=f'cache{i}');inputs.append(cache);joined=L.Concatenate(axis=1)([cache,x]);outputs[f'cache{i}_out']=L.Lambda(lambda z,h=h:z[:,-h:,:,:])(joined)
  y=L.DepthwiseConv2D((5,1),dilation_rate=(h//4,1),name=f'dw{i}')(joined);y=L.ReLU()(y);y=L.Conv2D(64,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([y,x]))
 outputs['logits']=L.Conv2D(classes,1,name='readout')(x);model=tf.keras.Model(inputs,outputs)
 def arr(k):return state[k].detach().cpu().numpy()
 for i in range(6):
  prefix=f'encoder.blocks.{i}.'
  model.get_layer(f'dw{i}').set_weights([arr(prefix+'dw.weight').transpose(2,0,1)[:,None,:,:],arr(prefix+'dw.bias')])
  model.get_layer(f'pw{i}').set_weights([arr(prefix+'pw.weight').transpose(2,1,0)[None,:,:,:],arr(prefix+'pw.bias')])
 model.get_layer('readout').set_weights([arr('readout.weight').transpose(2,1,0)[None,:,:,:],arr('readout.bias')]);return model

def main():
 torch.set_num_threads(1);tf.config.set_visible_devices([],'GPU');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 root=Path('.local/wakeword/causal-generator-unit-supervised-v1');cp=root/'final.pt';checkpoint=torch.load(cp,map_location='cpu',weights_only=False);assert checkpoint['step']==5000;state=checkpoint['model'];pt=CausalGeneratorUnitModel(34).cuda().eval();pt.load_state_dict(state);keras=build(state);rng=np.random.default_rng(726);cache=pt.encoder.initial_state(device='cuda');ks={f'cache{i}':np.zeros((1,h,1,64),np.float32) for i,h in enumerate([4,8,16,32,64,128])};maximum=0.;relative=0.
 with torch.no_grad():
  for _ in range(64):
   # Analytic parity only: actual audio calibration is a separate required step.
   x=rng.uniform(-10,10,(1,64,3)).astype(np.float32);y,cache=pt(torch.tensor(x,device='cuda'),cache);result=keras(dict(chunk=x.transpose(0,2,1)[:,:,None,:],**ks),training=False);expected=y.cpu().numpy().transpose(0,2,1)[:,:,None,:];actual=result['logits'].numpy();maximum=max(maximum,float(np.abs(actual-expected).max()));assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4)
   for i,c in enumerate(cache):
    ks[f'cache{i}']=result[f'cache{i}_out'].numpy();assert np.allclose(ks[f'cache{i}'],c.cpu().numpy().transpose(0,2,1)[:,:,None,:],atol=2e-4,rtol=2e-4)
 dest=root/'trained.keras';keras.save(dest)
 out=dict(schema='causal-unit-trained-transfer-v1',checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),keras_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),analytic_feedback_calls=64,max_abs_logit_error=maximum,all_caches_close=True,qualification=False,limitations='Trained weights transferred, analytic cache feedback comparison only. Not waveform recognition or integer conversion/calibration.')
 with Path('research/wakeword_p4/results/4090-causal-unit-trained-transfer-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
