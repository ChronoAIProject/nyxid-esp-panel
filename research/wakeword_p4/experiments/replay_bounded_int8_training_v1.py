"""All admitted training waves, frozen INT8 model/policy; no tuning."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ai_edge_litert.interpreter import Interpreter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_sampler_v1 import Sampler,INDEX
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from bounded_unit_event_policy_v1 import BoundedUnitEventPolicy
B=Path('.local/wakeword');R=Path('research/wakeword_p4/results');ROOT=B/'causal-generator-delay40-supervised-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(v,d):
 scale,zero=d['quantization'];return np.clip(np.rint(v/scale)+zero,-128,127).astype(np.int8)
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();sampler=Sampler();mapping=sampler.data['dense_to_original_label'];fp=B/'matchbox-grid-aligned-v1/frontend.pt';front=ContinuousPastMFCC(torch.load(fp,map_location='cpu',weights_only=True)).cuda().eval();mp=ROOT/'int8.tflite';assert sha(mp)=='21dbf17f03459c1fe213c83182d246594cde66f1e649ae8330d54b49bb85c0d7';it=Interpreter(model_path=str(mp),num_threads=1);it.allocate_tensors();runner=it.get_signature_runner();ins=runner.get_input_details();outs=runner.get_output_details();assert outs['probabilities']['quantization']==(1/256,-128)
 dest=ROOT/'bounded-all-training-v1.jsonl';records=[];started=time.monotonic()
 with dest.open('x',buffering=1) as log,torch.no_grad():
  for idx,r in enumerate(sampler.rows):
   assert r['role']=='training' and sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and pcm.ndim==1 and len(pcm)==r['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256'];clock=FrameClock();frames,ends=clock.push(pcm);features=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).cpu().numpy()[:,None,:];state={f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};policy=BoundedUnitEventPolicy();events=[];digest=hashlib.sha256();calls=0
   for start in range(0,len(features)-2,3):
    result=runner(chunk=quant(features[None,start:start+3],ins['chunk']),**state);codes=result['probabilities'][0,:,0,:].astype(np.int16)+128;digest.update(result['probabilities'].tobytes());calls+=1
    for j,row in enumerate(codes):
     label=int(row.argmax());original=mapping[label];token=original-1 if original else None
     if policy.push(token,int(row[label])):events.append(int(ends[start+j]))
    for i in range(6):
     scale,zero=outs[f'cache{i}_out']['quantization'];state[f'cache{i}']=quant((result[f'cache{i}_out'].astype(np.float64)-zero)*scale,ins[f'cache{i}'])
   row=dict(index=idx,class_id=r['class_id'],carrier=r['carrier'],source_sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=len(pcm),frames_consumed=calls*3,unscored_complete_frames=len(frames)-calls*3,clock=clock.finish(),event_samples=events,probability_code_sha256=digest.hexdigest());records.append(row);log.write(json.dumps(row)+'\n')
   if (idx+1)%100==0:print(json.dumps(dict(recordings=idx+1,elapsed_seconds=time.monotonic()-started)),flush=True)
 out=dict(schema='bounded-int8-all-training-v1',files={str(p):sha(p) for p in [Path(__file__),mp,fp,INDEX,Path('research/wakeword_p4/training/bounded_unit_event_policy_v1.py'),dest]},recordings=len(records),target_hits=sum(bool(r['event_samples']) for r in records if r['class_id']==1),target_count=720,negative_hits=sum(bool(r['event_samples']) for r in records if r['class_id']!=1),negative_count=3600,total_events=sum(len(r['event_samples']) for r in records),elapsed_seconds=time.monotonic()-started,records=records,qualification=False,limitations='Training-only all4320sources including24calibration records. Actual INT8 confidence128/two-frame uncertainty/span200/refractory200. Per-record reset, complete PCM, final0–2complete frames retained unscored without padding. No continuous-background FAR, fresh development, threshold selection or generalization claim. CUDA external float MFCC, CPU1thread integer neural runtime.')
 with (R/'4090-bounded-int8-all-training-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['records','files']}))
if __name__=='__main__':main()
