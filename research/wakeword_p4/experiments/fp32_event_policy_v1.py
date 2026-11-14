"""Raw FP32 diagnostic events: binary64 fsum, never UINT8 score rounding."""
import math,numbers
EVENT_ARITHMETIC='fp32-to-binary64-fsum-five-v1'
def event_steps(probabilities,threshold):
 if isinstance(threshold,bool) or not isinstance(threshold,numbers.Real) or not math.isfinite(float(threshold)) or not 0<float(threshold)<=1:
  raise ValueError('Expected finite threshold in (0,1]')
 threshold=float(threshold);events=[];window=[];last=-1000;armed=True
 for step,value in enumerate(probabilities,1):
  if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or not 0<=float(value)<=1:
   raise ValueError('Expected finite probability in [0,1]; codes are not probabilities')
  window.append(float(value))
  if len(window)>5:window.pop(0)
  if len(window)<5:continue
  mean=math.fsum(window)/5.0
  if mean<threshold*.5 and step-last>=67:armed=True
  if armed and mean>=threshold and step-last>=67:events.append(step);last=step;armed=False
 return events
