"""Enumerated probability oracle, deterministiccases and CUDAgradientfunction."""
import hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from ordered_unit_background_loss_v1 import acceptance_probability,background_rejection_loss
from ordered_generator_matcher_v1 import PATTERN,BOUNDARIES
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def symbolic(seq,pattern,boundaries):
 k=0
 for token in seq:
  if k and token==pattern[k-1]:continue
  if k==len(pattern) and token in boundaries:return True
  if k<len(pattern) and token==pattern[k]:k+=1
  else:k=1 if token==pattern[0] else 0
 return False
def main():
 rng=np.random.default_rng(726);cases=[]
 for frames in [3,4,5,6]:
  p=rng.uniform(.1,1,size=(4,frames));p/=p.sum(0);expected=0
  for seq in itertools.product(range(4),repeat=frames):
   if symbolic(seq,(1,2),(3,)):expected+=np.prod([p[token,t] for t,token in enumerate(seq)])
  actual=float(acceptance_probability(torch.tensor(p[None]),(1,2),(3,))[0]);assert abs(actual-expected)<1e-12;cases.append(dict(frames=frames,enumerated=expected,automaton=actual))
 dense={v:i+1 for i,v in enumerate(sorted(set(PATTERN)|set(BOUNDARIES)))}
 seq=[dense[t] for t in [*PATTERN,3]];one=torch.zeros(1,len(dense)+1,len(seq),dtype=torch.float64)
 for i,t in enumerate(seq):one[0,t,i]=1
 assert float(acceptance_probability(one,tuple(dense[t] for t in PATTERN),tuple(dense[t] for t in BOUNDARIES))[0])==1
 assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(726)
 logits=torch.randn(2,4,80,dtype=torch.float64,device='cuda',requires_grad=True);loss=background_rejection_loss(logits.softmax(1),(1,2),(3,));loss.backward();assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and bool((logits.grad!=0).any())
 out=dict(schema='ordered-background-loss-function-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/ordered_unit_background_loss_v1.py')]},enumeration_cases=cases,cuda_loss=float(loss.detach()),all_gradients_finite_nonzero=True,optimizer_updates=0,real_background_transcripts_created=0,policy='Analyticinputsonly, noaudio/recognition/training. Differentiableautomaton modelsindependentcategoricaldraws; repeatlastunit/nextunit/reset/startrestart plusboundary andabsorbingaccept. NOTfullKMPfallback or deployedconfidence/refractory/spanpolicy; actualgreedymatcher andautomaton have differenterrorfunctions. Surrogatenonwakelossforuntranscribedbackground, nottrueprobabilityofwakeininferenceorphoneticlabels. Clampnearone onlynumericaloverflowguard; longbackgroundlossmay saturate, realgradientscale/supportmustmeasurebeforetraining. Doesnotclaim this solvespriornegativepeak/temporalriskfailures. v30unchanged.')
 with (R/'4090-ordered-background-loss-function-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
