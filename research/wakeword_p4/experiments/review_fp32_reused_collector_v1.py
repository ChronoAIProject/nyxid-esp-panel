"""Constructed complete input/clock/fresh-interpreter tests; no real model/media."""
import hashlib,json
from pathlib import Path
import numpy as np
from collect_fp32_reused_development_v1 import prepare,predict
class Fake:
 count=0
 def __init__(self,**kwargs):self.calls=0;Fake.count+=1
 def allocate_tensors(self):pass
 def get_input_details(self):return [dict(index=0,dtype=np.float32,shape=np.array([1,3,40]))]
 def get_output_details(self):return [dict(index=1,dtype=np.float32,shape=np.array([1,1]))]
 def set_tensor(self,index,value):assert index==0 and value.shape==(1,3,40);self.value=value.copy()
 def invoke(self):self.calls+=1
 def get_tensor(self,index):return np.asarray([[self.value[0,-1,0]/26]],np.float32)
def main():
 raw=np.arange(320,dtype=np.uint16).reshape(8,40);values,support,clocks=prepare(raw,1600)
 assert values.shape==(9,40) and np.array_equal(support,[800,1280,1760]) and np.array_equal(clocks,[800,1280,1600]) and np.all(values[-1]==0)
 a=predict(values,'constructed',Fake);b=predict(values,'constructed',Fake)
 assert Fake.count==2 and a.shape==(3,) and a.tobytes()==b.tobytes() and a[-1]==0 and a[0]>0
 rejected=0
 for raw,samples in [(raw.astype(np.float32),1600),(raw[:-1],1600),(raw,0)]:
  try:prepare(raw,samples)
  except ValueError:rejected+=1
  else:raise AssertionError('Invalid frontend accepted')
 class Nonfinite(Fake):
  def get_tensor(self,index):return np.array([[np.nan]],np.float32)
 try:predict(values,'constructed',Nonfinite)
 except ValueError:rejected+=1
 else:raise AssertionError('NaN accepted')
 p=Path(__file__).with_name('collect_fp32_reused_development_v1.py')
 Path('research/wakeword_p4/results/4090-fp32-reused-collector-constructed-v1.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),complete_tail_and_clocks=True,fresh_interpreters=True,startup_retained=True,invalid_checks=rejected,all_passed=True,limits='Fake interpreter/constructed arrays only; no actual inference, feature extraction or media.'),indent=2)+'\n');print('Constructed collector checks passed')
if __name__=='__main__':main()
