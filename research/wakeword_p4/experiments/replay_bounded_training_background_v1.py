"""Continuous concatenated admitted training backgrounds, single reset."""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ai_edge_litert.interpreter import Interpreter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from causal_unit_sampler_v1 import Sampler
from bounded_unit_event_policy_v1 import BoundedUnitEventPolicy
B=Path('.local/wakeword');R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(v,d):
 scale,zero=d['quantization'];return np.clip(np.rint(v/scale)+zero,-128,127).astype(np.int8)
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();source=B/'matchbox-long-material-v1/records.json';records=[r for r in json.load(source.open()) if r['kind']=='libri-training-background'];assert len(records)==1004 and all(r['role']=='training' for r in records);mapping=Sampler().data['dense_to_original_label'];fp=B/'matchbox-grid-aligned-v1/frontend.pt';front=ContinuousPastMFCC(torch.load(fp,map_location='cpu',weights_only=True)).cuda().eval();mp=B/'causal-generator-delay40-supervised-v1/int8.tflite';assert sha(mp)=='21dbf17f03459c1fe213c83182d246594cde66f1e649ae8330d54b49bb85c0d7';it=Interpreter(model_path=str(mp),num_threads=1);it.allocate_tensors();runner=it.get_signature_runner();ins=runner.get_input_details();outs=runner.get_output_details();assert outs['probabilities']['quantization']==(1/256,-128);state={f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};policy=BoundedUnitEventPolicy();clock=FrameClock();pending=np.empty((0,1,64),np.float32);times=np.empty(0,np.int64);events=[];sources=[];digest=hashlib.sha256();calls=0;started=time.monotonic()
 with torch.no_grad():
  for idx,r in enumerate(records):
   assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and pcm.ndim==1 and len(pcm)==r['samples'];before=clock.received;frames,ends=clock.push(pcm);features=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).cpu().numpy()[:,None,:];joined=np.concatenate([pending,features]);t=np.concatenate([times,ends]);n=len(joined)//3*3
   for start in range(0,n,3):
    result=runner(chunk=quant(joined[None,start:start+3],ins['chunk']),**state);codes=result['probabilities'][0,:,0,:].astype(np.int16)+128;digest.update(result['probabilities'].tobytes());calls+=1
    for j,row in enumerate(codes):
     label=int(row.argmax());original=mapping[label];token=original-1 if original else None
     if policy.push(token,int(row[label])):events.append(int(t[start+j]))
    for i in range(6):
     scale,zero=outs[f'cache{i}_out']['quantization'];state[f'cache{i}']=quant((result[f'cache{i}_out'].astype(np.float64)-zero)*scale,ins[f'cache{i}'])
   pending=joined[n:].copy();times=t[n:].copy();sources.append(dict(path=r['path'],sha256=r['sha256'],pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),global_start_sample=before,global_end_sample=clock.received))
   if (idx+1)%100==0:print(json.dumps(dict(sources=idx+1,calls=calls,events=len(events),elapsed_seconds=time.monotonic()-started)),flush=True)
 hours=clock.received/16000/3600;scored_end=int((calls*3-1)*160+400) if calls else 0;scored_hours=scored_end/16000/3600
 out=dict(schema='bounded-int8-continuous-training-background-v1',files={str(p):sha(p) for p in [Path(__file__),source,fp,mp,Path('research/wakeword_p4/training/bounded_unit_event_policy_v1.py')]},sources=sources,clock=clock.finish(),unscored_complete_frames=len(pending),neural_calls=calls,model_resets=1,frontend_resets=1,policy_resets=1,synthetic_join_boundaries=len(sources)-1,events_samples=events,event_count=len(events),input_hours=hours,scored_prefix_hours=scored_hours,zero_event_poisson_upper95_per_hour=-math.log(.05)/scored_hours if not events else None,probability_code_sha256=digest.hexdigest(),elapsed_seconds=time.monotonic()-started,qualification=False,limitations='Consumed training-only Libri audio, concatenated in admitted metadata order without inserted silence; all PCM/hashes checked, residual PCM/feature cache and neural/policy state preserved across joins. Artificial joins are not natural uninterrupted recordings. No transcripts used, no padding, no threshold tuning or independent FAR evidence. Poisson bound is descriptive for this reused artificial stream only;3.57hours cannot support<=0.5/h even if zero events. CUDAfloat MFCC, CPU1threadINT8.')
 with (R/'4090-bounded-int8-continuous-training-background-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['files','sources']}))
if __name__=='__main__':main()
