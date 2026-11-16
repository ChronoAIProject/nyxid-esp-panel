"""Differentiable exact-prefix automaton probability, no background transcript.

Contiguous token draws from per-frame categoricaldistribution; adjacentrepeats
allowed per matchedunit, targetfollowedbyboundary. Otherunit resets exceptstart.
Absorbing accept means probabilityofatleastonepattern in complete recording.
This softautomaton is a surrogate, not the confidence/cooldown greedy decoder.
"""
import torch

def acceptance_probability(probabilities,pattern,boundaries):
 if probabilities.ndim!=3:raise ValueError('Expected[B,C,T]')
 if not pattern or len(set(pattern))!=len(pattern):
  # Generic repeated nonadjacentpattern supported below; adjacentduplicate not.
  if not pattern or any(a==b for a,b in zip(pattern,pattern[1:])):raise ValueError('Empty/adjacentduplicatepattern')
 if pattern[-1] in boundaries:raise ValueError('Terminal unit cannot also be a boundary')
 batch,classes,frames=probabilities.shape
 if max([*pattern,*boundaries])>=classes:raise ValueError('Unknownclass')
 n=len(pattern);state=probabilities.new_zeros(batch,n+2);state[:,0]=1
 for t in range(frames):
  p=probabilities[:,:,t];parts=[probabilities.new_zeros(batch) for _ in range(n+2)];parts[-1]=state[:,-1]
  # Statek matchedfirstk units; repeatedlastunit waits, nextadvances.
  for k in range(n+1):
   mass=state[:,k];used=probabilities.new_zeros(batch)
   if k:
    stay=p[:,pattern[k-1]];parts[k]=parts[k]+mass*stay;used=used+stay
   if k<n:
    advance=p[:,pattern[k]];parts[k+1]=parts[k+1]+mass*advance;used=used+advance
   else:
    advance=p[:,list(boundaries)].sum(1);parts[-1]=parts[-1]+mass*advance;used=used+advance
   if k and pattern[0]!=pattern[k-1]:
    restart=p[:,pattern[0]];parts[1]=parts[1]+mass*restart;used=used+restart
   parts[0]=parts[0]+mass*(1-used)
  state=torch.stack(parts,dim=1)
 return state[:,-1]

def background_rejection_loss(probabilities,pattern,boundaries):
 q=acceptance_probability(probabilities,pattern,boundaries)
 # Float64 recommended tokeep smallsequenceprobabilities/gradients.
 return -torch.log1p(-q.clamp(max=1-torch.finfo(q.dtype).eps)).mean()
