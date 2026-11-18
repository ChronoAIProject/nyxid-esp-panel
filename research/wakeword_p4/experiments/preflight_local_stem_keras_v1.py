"""Zero-update Torch/Keras local stem parity and pinned frontend tone-order probe."""
import os
os.environ.update(TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,json,hashlib,inspect
from pathlib import Path
import numpy as np
import torch,tensorflow as tf
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_local_stem_v1 import Binary40LocalStem
from causal_binary40_span_v1 import CONTROL
from build_binary40_local_stem_keras_v1 import build
from features import features
import micro_features_cpp
from pymicro_features import MicroFrontend
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tf.config.set_visible_devices([],'GPU')
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
m=Binary40LocalStem().cuda().eval();keras,h,c=build(m.state_dict(),CONTROL);rng=np.random.default_rng(726)
st=m.initial_state(device='cuda');ks={f'cache{i}':np.zeros((1,a,1,b),np.float32) for i,(a,b) in enumerate(zip(h,c))};pe=ce=0.
with torch.inference_mode():
 for call in range(64):
  x=rng.uniform(0,26,(1,40,3)).astype(np.float32);y,st=m(torch.from_numpy(x).cuda(),st);res=keras(dict(chunk=x.transpose(0,2,1)[:,:,None,:],**ks),training=False)
  pe=max(pe,float(np.abs(res['probability'].numpy().reshape(-1)-torch.sigmoid(y).cpu().numpy().reshape(-1)).max()))
  for i,v in enumerate(st):
   ks[f'cache{i}']=res[f'cache{i}_out'].numpy();expected=v.cpu().numpy().transpose(0,2,1)[:,:,None,:];ce=max(ce,float(np.abs(ks[f'cache{i}']-expected).max()));assert np.allclose(ks[f'cache{i}'],expected,atol=2e-4,rtol=2e-4)
assert pe<2e-4
tones=[]
for hz in [250,500,1000,2000,4000]:
 pcm=np.rint(5000*np.sin(2*np.pi*hz*np.arange(16000)/16000));a=features(pcm,MicroFrontend());mean=a[:20].mean(axis=0);tones.append(dict(hz=hz,peak_channel=int(mean.argmax())))
assert all(a['peak_channel']<b['peak_channel'] for a,b in zip(tones,tones[1:]))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=Path('.local/wakeword/frontend-coefficients-source-v1/pymicro_features-2.0.2/src/micro_features.cpp')
out=dict(gpu=torch.cuda.get_device_name(),optimizer_updates=0,calls=64,probability_max_error=pe,cache_max_error=ce,tone_probe=tones,frontend_binary=dict(path=micro_features_cpp.__file__,sha256=sha(micro_features_cpp.__file__)),source_config=dict(path=str(src),sha256=sha(src),channels=40,lower_hz=125,upper_hz=7500,pcan=True,log=True),limits='Installed frontend ascending tone peak order plus cached source configuration; not a full independent frontend binary/source equivalence proof. Random input Keras parity, no speech recognition/INT8/P4 qualification.')
with open('research/wakeword_p4/results/4090-local-stem-keras-frontend-preflight-v1.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
