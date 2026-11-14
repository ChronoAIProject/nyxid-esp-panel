"""Explicit streaming readiness policy; does not modify any model or old evaluator."""
import math
import numbers
from uint8_event_policy_v1 import event_steps

class WakeReadiness:
    def __init__(self, threshold, context_frames=309, stride_frames=3):
        if not isinstance(threshold,numbers.Real) or isinstance(threshold,bool) or not math.isfinite(threshold) or not 0<threshold<=1:
            raise ValueError('Invalid threshold')
        if type(context_frames) is not int or context_frames<=0 or type(stride_frames) is not int or stride_frames<=0:
            raise ValueError('Invalid model geometry')
        self.first_valid_call=(context_frames+stride_frames-1)//stride_frames
        self.first_ready_call=self.first_valid_call+4
        self.threshold=float(threshold);self.call=0;self.window=[];self.total=0
        self.last=-1000;self.armed=True
    @property
    def ready(self):return self.call>=self.first_ready_call
    def push(self,code):
        if isinstance(code,bool) or not isinstance(code,numbers.Integral) or not 0<=code<=255:
            raise ValueError('Expected UINT8 output')
        self.call+=1
        if self.call<self.first_valid_call:return False
        self.window.append(int(code));self.total+=int(code)
        if len(self.window)>5:self.total-=self.window.pop(0)
        if len(self.window)<5:return False
        mean=self.total/1280.0
        if mean<self.threshold*.5 and self.call-self.last>=67:self.armed=True
        if self.armed and mean>=self.threshold and self.call-self.last>=67:
            self.last=self.call;self.armed=False;return True
        return False

def ready_event_steps(raw,threshold,context_frames=309,stride_frames=3):
    # Equivalent bulk application; state begins at the first valid prediction.
    skip=(context_frames+stride_frames-1)//stride_frames-1
    return [s+skip for s in event_steps(raw[skip:],threshold)]
