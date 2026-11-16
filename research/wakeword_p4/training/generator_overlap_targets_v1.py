"""Declared synthetic allocation mixture, not acoustic posterior or truephones."""
import numpy as np
def targets(samples,lead,intervals,delays=(0,),amplitudes=(1.,)):
 if samples<400 or lead<0:raise ValueError('Complete frame required')
 if len(delays)!=len(amplitudes) or not delays or min(delays)<0:raise ValueError('Invalid RIR')
 assert intervals and intervals[0]['start_sample']==0
 for a,b in zip(intervals,intervals[1:]):assert a['end_sample']==b['start_sample']
 assert lead+intervals[-1]['end_sample']+max(delays)<=samples
 ends=np.arange(400,samples+1,160,dtype=np.int64);starts=ends-400
 size=max(t['token_id'] for t in intervals)+2;result=np.zeros((len(ends),size),np.float64);weights=np.abs(np.asarray(amplitudes,np.float64));assert weights.sum()>0;weights/=weights.sum();position_mass=np.zeros(len(intervals),np.float64)
 for delay,weight in zip(delays,weights,strict=True):
  for i,token in enumerate(intervals):
   lo=lead+token['start_sample']+delay;hi=lead+token['end_sample']+delay;mass=np.maximum(0,np.minimum(ends,hi)-np.maximum(starts,lo))/400*weight
   result[:,token['token_id']+1]+=mass;position_mass[i]+=mass.sum()
 result[:,0]=1-result[:,1:].sum(axis=1);assert result.min()>-1e-12 and np.allclose(result.sum(axis=1),1,atol=1e-12)
 return np.maximum(result,0).astype(np.float32),ends,position_mass
