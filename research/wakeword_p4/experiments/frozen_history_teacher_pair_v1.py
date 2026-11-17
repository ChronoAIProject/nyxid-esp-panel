"""Fixed physical pair cache and loss components, no optimizer or policy."""
import gzip,json
from pathlib import Path
import numpy as np
import torch
R=Path('research/wakeword_p4/results')
class Pairs:
 def __init__(self):
  self.meta=json.loads((R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json').read_text())['inventory'];self.cache=json.loads((R/'4090-frozen-history-teacher-cache-original-v1.json').read_text());self.rows=self.cache['records']
  self.arrays={h:np.memmap(f['path'],dtype='<u2',mode='r').reshape(-1,40) for h,f in self.cache['files'].items()}
  with gzip.open(R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz','rt') as f:self.schedule=[json.loads(s) for s in f]
 def batch(self,indices):
  frames=max(self.meta[i]['frames'] for i in indices);x=np.zeros((len(indices)*2,frames,40),np.float32);pw=np.zeros((len(x),frames//3),np.float32);nw=np.zeros_like(pw);tw=np.zeros((len(indices),frames//3),np.float32)
  for j,i in enumerate(indices):
   r=self.meta[i];c=self.rows[i];assert c['inventory_index']==i and c['source_id']==r['source']['id'];n=r['frames'];support=(3*np.arange(1,n//3+1)+2)*160;end=r['prefix_samples']+r['reference_samples'];label=r['source']['label']
   positive=(support>=end+480)&(support<=end+4320) if label else np.zeros(len(support),bool);negative=support<=r['prefix_samples'] if label else np.ones(len(support),bool);transfer=(positive if label else negative)&(support>r['prefix_samples']);assert transfer.any()
   tw[j,:len(support)]=transfer/transfer.sum()
   for view,h in enumerate(['silence','speech']):
    at=2*j+view;raw=self.arrays[h][c['offset_frames']:c['offset_frames']+n];x[at,:n]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);pw[at,:len(support)]=positive/max(1,positive.sum());nw[at,:len(support)]=negative/negative.sum()
  return x,pw,nw,tw

def tensor(a):return torch.from_numpy(a).cuda()
def bce(logits,pw,nw):return ((torch.nn.functional.softplus(-logits)*pw+20*torch.nn.functional.softplus(logits)*nw).sum(1)).mean()
def pair_components(student,teacher,batch):
 x,p,n,w=batch;xt=tensor(x).permute(0,2,1);z,_=student(xt,student.initial_state(len(x),'cuda'));z=z[:,0]
 with torch.no_grad():t,_=teacher(xt[::2],teacher.initial_state(len(x)//2,'cuda'));t=t[:,0];q=torch.sigmoid(t)
 supervised=bce(z,tensor(p),tensor(n));kl=((torch.nn.functional.softplus(z[1::2])-q*z[1::2]-(torch.nn.functional.softplus(t)-q*t))*tensor(w)).sum(1).mean()
 return supervised,kl
