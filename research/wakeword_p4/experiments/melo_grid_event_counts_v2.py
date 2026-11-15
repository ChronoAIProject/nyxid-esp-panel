"""Exact integer form of the predeclared five-UINT8 grid event policy."""
import numpy as np
from numba import njit

@njit(cache=True)
def _counts(raw,total_samples,lead_samples):
 result=np.zeros((1280,3),np.int64)
 for index in range(1280):
  k=index+1;window_sum=0;last=-1000;armed=True
  for j in range(len(raw)):
   step=j+1;window_sum+=int(raw[j])
   if j>=5:window_sum-=int(raw[j-5])
   if j<4:continue
   if 2*window_sum<k and step-last>=67:armed=True
   if armed and window_sum>=k and step-last>=67:
    last=step;armed=False
    support=(3*step+2)*160
    clock=min(((support+479)//480)*480,total_samples+81600)-81600
    if clock<=0:result[index,2]+=1
    else:
     result[index,0]+=1
     if clock>=lead_samples:result[index,1]=1
 return result

def grid_event_counts(raw,total_samples,lead_samples):
 if raw.dtype!=np.uint8 or raw.ndim!=1:raise ValueError('Complete UINT8 vector required')
 if type(total_samples)is not int or total_samples<=0 or type(lead_samples)is not int or not 0<=lead_samples<total_samples:raise ValueError('Invalid whole source clock')
 return _counts(raw,total_samples,lead_samples)
