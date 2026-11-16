"""Complete recording batch; padded features/targets have zero loss mass."""
import hashlib
import numpy as np
import soundfile as sf
import torch
from continuous_past_mfcc_v1 import FrameClock
from generator_overlap_targets_v1 import targets

def batch(sampler,frontend):
 descriptors=sampler.descriptors();features=[];probabilities=[];lengths=[]
 labels=sampler.data['dense_to_original_label']
 for d in descriptors:
  r=sampler.rows[d['index']]
  with open(r['path'],'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
  pcm,rate=sf.read(r['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
  clock=FrameClock();frames,ends=clock.push(pcm);kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}
  y,t,mass=targets(r['samples'],r['lead'],r['intervals'],**kwargs);assert np.array_equal(ends,t) and (mass>0).all()
  dense=np.zeros((len(y),len(labels)),np.float32)
  for i,label in enumerate(labels):
   if label<y.shape[1]:dense[:,i]=y[:,label]
  assert np.allclose(dense.sum(1),1,atol=1e-6)
  with torch.no_grad():f=frontend(torch.from_numpy(frames.astype(np.float32)/32768).cuda()).T
  features.append(f);probabilities.append(torch.from_numpy(dense.T).cuda());lengths.append(len(frames))
  d['frames']=len(frames);d['finish_residual_samples']=clock.finish()['residual_samples']
 maximum=max(lengths);x=torch.zeros(len(lengths),64,maximum,device='cuda');p=torch.zeros(len(lengths),len(labels),maximum,device='cuda');mask=torch.zeros(len(lengths),maximum,dtype=torch.bool,device='cuda')
 for i,n in enumerate(lengths):x[i,:,:n]=features[i];p[i,:,:n]=probabilities[i];mask[i,:n]=True
 return x,p,mask,descriptors

def recording_normalized_loss(logits,probabilities,mask):
 if logits.shape!=probabilities.shape or mask.shape!=(logits.shape[0],logits.shape[2]):raise ValueError('Mismatched batch geometry')
 lengths=mask.sum(1);assert bool((lengths>0).all())
 perframe=-(probabilities*torch.log_softmax(logits,dim=1)).sum(1)
 return ((perframe*mask).sum(1)/lengths).mean()
