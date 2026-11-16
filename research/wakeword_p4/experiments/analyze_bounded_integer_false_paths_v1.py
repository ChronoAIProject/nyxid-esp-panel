"""All26known training false recordings, actual integer unit/interval diagnosis."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ai_edge_litert.interpreter import Interpreter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_sampler_v1 import INDEX
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from bounded_unit_event_policy_v1 import BoundedUnitEventPolicy
from generator_overlap_targets_v1 import targets
B=Path('.local/wakeword');R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(v,d):
 scale,zero=d['quantization'];return np.clip(np.rint(v/scale)+zero,-128,127).astype(np.int8)
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();data=json.load(gzip.open(INDEX));mapping=data['dense_to_original_label'];census=R/'4090-bounded-int8-all-training-v1.json';selected=[r for r in json.load(census.open())['records'] if r['class_id']!=1 and r['event_samples']];assert len(selected)==26
 fp=B/'matchbox-grid-aligned-v1/frontend.pt';front=ContinuousPastMFCC(torch.load(fp,weights_only=True,map_location='cpu')).cuda().eval();mp=B/'causal-generator-delay40-supervised-v1/int8.tflite';it=Interpreter(model_path=str(mp),num_threads=1);it.allocate_tensors();run=it.get_signature_runner();ins=run.get_input_details();outs=run.get_output_details();results=[]
 with torch.no_grad():
  for original in selected:
   r=data['records'][original['index']];assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];clock=FrameClock();frames,ends=clock.push(pcm);f=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).cpu().numpy()[:,None,:];state={f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};parts=[];policy=BoundedUnitEventPolicy();events=[]
   for start in range(0,len(f)-2,3):
    output=run(chunk=quant(f[None,start:start+3],ins['chunk']),**state);codes=output['probabilities'][0,:,0,:].astype(np.int16)+128;parts.append(codes)
    for j,row in enumerate(codes):
     label=int(row.argmax());token=mapping[label]-1 if mapping[label] else None
     if policy.push(token,int(row[label])):events.append(int(ends[start+j]))
    for i in range(6):
     scale,zero=outs[f'cache{i}_out']['quantization'];state[f'cache{i}']=quant((output[f'cache{i}_out'].astype(np.float64)-zero)*scale,ins[f'cache{i}'])
   assert events==original['event_samples'];codes=np.concatenate(parts);labels=codes.argmax(1);tokens=np.array([mapping[int(v)]-1 if mapping[int(v)] else -1 for v in labels]);analysis=[]
   for position,t in enumerate(r['intervals']):
    # Output labels delayed40ms; descriptive centers aligned to original allocation.
    centers=ends[:len(codes)]-200-640;mask=(centers>=r['lead']+t['start_sample'])&(centers<r['lead']+t['end_sample']);values,counts=np.unique(tokens[mask],return_counts=True);expected=mapping.index(t['token_id']+1)
    analysis.append(dict(position=position,token=t['token_id'],start_sample=r['lead']+t['start_sample'],end_sample=r['lead']+t['end_sample'],center_frames=int(mask.sum()),predicted_counts={str(int(v)):int(n) for v,n in zip(values,counts)},mean_expected_code=float(codes[mask,expected].mean()) if mask.any() else None))
   event_windows=[]
   for event in events:
    mask=(ends[:len(codes)]>=event-32000)&(ends[:len(codes)]<=event);event_windows.append(dict(event_sample=event,frame_ends=ends[:len(codes)][mask].tolist(),predicted_tokens=tokens[mask].tolist(),confidence_codes=codes.max(1)[mask].tolist()))
   results.append(dict(index=original['index'],class_id=r['class_id'],carrier=r['carrier'],variant=r['variant'],source_sha256=r['sha256'],events_samples=events,interval_analysis=analysis,event_windows=event_windows))
 out=dict(schema='bounded-integer-false-paths-v1',files={str(p):sha(p) for p in [Path(__file__),INDEX,census,fp,mp]},records=results,integer_census_events_exactly_reproduced=True,optimizer_updates=0,limitations='All26known false training recordings, descriptive generator-center alignment with40msoutput delay. Not acoustic phone truth; rectangular centers omit RIR energy/phase. Actual integer codes and full pre-event2second windows retained. No independent evaluation, threshold or policy change.')
 with (R/'4090-bounded-integer-false-paths-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print('Reproduced',len(results),'false recordings')
if __name__=='__main__':main()
