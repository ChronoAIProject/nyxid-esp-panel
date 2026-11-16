"""Training-only frontend function check, no neural model or heldout calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from continuous_past_mfcc_v1 import FrameClock,ContinuousPastMFCC
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import soundfile as sf
 material=B/'matchbox-long-material-v1/records.json';row=next(r for r in json.loads(material.read_text()) if r.get('engine')=='speecht5' and r['class_id']==1);assert row['role']=='training' and sha(row['path'])==row['sha256']
 pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
 # Exact index oracle for irregularchunks and partial ends, independent direct slices.
 sequences=[np.arange(n,dtype=np.int64).astype(np.int16) for n in [0,1,159,160,399,400,401,559,560,561,44800]]+[pcm]
 receipts=[]
 for raw in sequences:
  expected=np.array([raw[s:s+400] for s in range(0,max(0,len(raw)-399),160)],dtype=np.int16).reshape(-1,400)
  ends=np.arange(len(expected))*160+400
  for chunks in [[1,159,7,480,1024,3],[480],[160],[4096]]:
   c=FrameClock();parts=[];timestamps=[];offset=0;k=0
   while offset<len(raw):
    n=chunks[k%len(chunks)];f,t=c.push(raw[offset:offset+n]);parts.append(f);timestamps.append(t);offset+=n;k+=1
   actual=np.concatenate(parts) if parts else np.empty((0,400),np.int16);times=np.concatenate(timestamps) if timestamps else np.empty(0,np.int64)
   assert np.array_equal(actual,expected) and np.array_equal(times,ends);assert c.finish()['residual_samples']<400
  receipts.append(dict(samples=len(raw),frames=len(expected),residual=len(raw)-len(expected)*160))
 assert torch.cuda.is_available();torch.set_num_threads(1);statepath=B/'matchbox-grid-aligned-v1/frontend.pt';state=torch.load(statepath,map_location='cpu',weights_only=True);front=ContinuousPastMFCC(state).cuda().eval()
 frames,_=FrameClock().push(pcm);h=hashlib.sha256();count=0
 with torch.no_grad():
  for start in range(0,len(frames),32):
   x=torch.from_numpy(frames[start:start+32].astype(np.float32)/32768).cuda();out=front(x);assert out.shape==(len(x),64) and torch.isfinite(out).all();h.update(out.cpu().numpy().tobytes());count+=len(x)
 result=dict(schema='continuous-past-mfcc-function-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/continuous_past_mfcc_v1.py'),material,statepath,Path(row['path'])]},gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,index_oracle_cases=receipts,chunk_schedules_per_case=4,all_raw_frame_indices_and_timestamps_passed=True,training_pcm_frames=count,frontend_output_sha256=h.hexdigest(),max_residual_pcm16_bytes=798,frame_acquisition_samples=400,hop_samples=160,future_samples=0,finish_padding=False,neural_calls=0,policy='Newfrontendproposal only. Raw index/chunk checks and finiteCUDAfeatures, no neuralCPU/GPU equality gate, recognition, numerical equivalence to historical frontend, calibration, P4port or promotion. Frame at400samples needs25msinitialacquisition and onlypastPCM.512FFTzeroextension and400Hann inheritbuffer values but continuousframe semantics differ from frozenwindowreflectMFCC. Incomingchunk memory and outputbatch excluded from residualbound; FFT/mel/DCT remainfloat. Full training recording loaded for diagnostic only; no heldout access.')
 with (R/'4090-continuous-past-mfcc-function-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,training_pcm_frames=count,gpu=result['gpu'])))
if __name__=='__main__':main()
