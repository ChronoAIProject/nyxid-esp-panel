"""Fixed causal acoustic targets plus unmodified Binary40 head contexts."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from frozen_history_teacher_pair_v1 import Pairs,tensor,bce
R=Path('research/wakeword_p4/results')
class Targets:
 def __init__(self):
  self.pairs=Pairs();self.report=json.loads((R/'4090-external-representation-target-cache-original-v1.json').read_text());self.rows=self.report['records']
  review=json.loads((R/'4090-external-representation-target-review-v1.json').read_text());assert review['all_passed'] and review['verified_sources']==4320
  with np.load(self.report['arrays']['path'],allow_pickle=False) as z:raw=z['raw_prefix'].astype(np.float64);mean=z['training_mean'].copy()
  delta=raw-mean;targets=(delta/np.linalg.norm(delta,axis=1,keepdims=True)).astype(np.float32)
  for i,(r,m,v) in enumerate(zip(self.rows,self.pairs.meta,targets,strict=True)):
   assert r['inventory_index']==i and r['source_id']==m['source']['id'] and r['label']==m['source']['label'] and r['end_sample']==m['reference_samples'] and r['prefix_samples']==m['prefix_samples'] and hashlib.sha256(v.tobytes()).hexdigest()==r['normalized_target_sha256']
  self.targets=targets;self.gpu_targets=tensor(targets).detach();self.anchors=np.asarray([r['student_anchor_zero_based_call'] for r in self.rows],np.int64)
 def batch(self,indices):
  physical=self.pairs.batch(indices);targets=np.repeat(self.targets[indices],2,axis=0);anchors=np.repeat(self.anchors[indices],2)
  assert all(self.rows[i]['student_anchor_release_sample']>=self.rows[i]['prefix_samples']+self.rows[i]['end_sample'] for i in indices)
  return physical,targets,anchors

def context_forward(model,x,states):
 captured=[]
 def hook(module,args):captured.append(args[0])
 handle=model.head.register_forward_pre_hook(hook)
 try:logits,updated=model(x,states)
 finally:handle.remove()
 assert len(captured)==1;joined=captured[0];batch,channels,_=joined.shape
 # Conv1d weights flatten in channel then two kernel taps, not tap then channel.
 contexts=joined.unfold(2,2,1).permute(0,1,3,2).reshape(batch,channels*2,-1)
 assert contexts.shape[2]==logits.shape[2]
 return logits,updated,contexts

def components(model,projection,batch):
 (x,p,n,w),target,anchors=batch
 z,_,contexts=context_forward(model,tensor(x).permute(0,2,1),model.initial_state(len(x),'cuda'))
 rows=torch.arange(len(x),device='cuda');selected=contexts[rows,:,torch.as_tensor(anchors,device='cuda')]
 predicted=projection(selected);target=tensor(target).detach()
 auxiliary=(1-torch.nn.functional.cosine_similarity(predicted,target,dim=1,eps=1e-12)).mean()
 return bce(z[:,0],tensor(p),tensor(n)),auxiliary,z[:,0],selected,predicted,target
