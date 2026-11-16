"""Training-only allocation/path error analysis, no parameter updates."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from generator_overlap_targets_v1 import targets
from ordered_generator_matcher_v1 import OrderedMatcher
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/causal-generator-unit-supervised-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spans(values,ends):
 out=[]
 for i,v in enumerate(values):
  if not out or out[-1]['token']!=v:out.append(dict(token=v,first_frame_end=int(ends[i]),last_frame_end=int(ends[i]),frames=1))
  else:out[-1]['last_frame_end']=int(ends[i]);out[-1]['frames']+=1
 return out

def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 index=R/'4090-causal-unit-training-index-v1.json.gz';data=json.load(gzip.open(index));mapping=data['dense_to_original_label'];cp=B/'final.pt';state=torch.load(cp,map_location='cpu',weights_only=False);model=CausalGeneratorUnitModel(len(mapping)).cuda().eval();model.load_state_dict(state['model']);front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval()
 diag=json.load((R/'4090-causal-unit-training-wave-diagnostic-v1.json').open());selected=[x for x in diag['records'] if (x['class_id']==1 and not x['events_samples']) or (x['class_id']!=1 and x['events_samples'])];results=[]
 with torch.no_grad():
  for d in selected:
   r=data['records'][d['index']];assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];clock=FrameClock();frames,ends=clock.push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None];logits,_=model(x,model.encoder.initial_state(device='cuda'));prob=logits.softmax(1)[0].T.cpu().numpy();pred=prob.argmax(1)
   kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {};q,t,mass=targets(r['samples'],r['lead'],r['intervals'],**kwargs);assert np.array_equal(t,ends)
   expected=q.argmax(1);tokens=[mapping[v]-1 if mapping[v] else None for v in pred];expected_tokens=[int(v)-1 if v else None for v in expected];matcher=OrderedMatcher();expected_events=[]
   for i,v in enumerate(expected_tokens):
    if v is None:matcher=OrderedMatcher()
    elif matcher.push(v):expected_events.append(int(ends[i]))
   allocation=[]
   for interval in r['intervals']:
    lo=r['lead']+interval['start_sample'];hi=r['lead']+interval['end_sample'];mask=(ends-200>=lo)&(ends-200<hi);counts={str(mapping[int(v)]-1 if mapping[int(v)] else None):int(n) for v,n in zip(*np.unique(pred[mask],return_counts=True))};label=mapping.index(interval['token_id']+1)
    allocation.append(dict(**interval,center_frames=int(mask.sum()),predicted_token_counts=counts,mean_expected_unit_probability=float(prob[mask,label].mean()) if mask.any() else None))
   results.append(dict(index=d['index'],path=r['path'],sha256=r['sha256'],class_id=r['class_id'],carrier=r['carrier'],variant=r['variant'],lead=r['lead'],predicted_events=d['events_samples'],allocation_argmax_events=expected_events,allocation_argmax_spans=spans(expected_tokens,ends),predicted_spans=spans(tokens,ends),interval_analysis=allocation))
 out=dict(schema='causal-unit-path-analysis-v1',files={str(p):sha(p) for p in [Path(__file__),index,cp]},selection='All misses and false events from fixed training batch24 diagnostic; descriptive analysis only, not independent evaluation.',records=results,optimizer_updates=0,limitations='Generator intervals/rectangular-overlap targets are synthetic allocation references, not acoustic phone truth. Center-frame statistics are descriptive and ignore phase/RIR energy interaction.')
 with (R/'4090-causal-unit-path-analysis-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 for r in results:
  print(r['carrier'],r['class_id'],'expected_events',r['allocation_argmax_events'])
  for a in r['interval_analysis']:
   if a['token_id'] in [50,156,47,102,16,56,53,61,55]:print(a['token_id'],a['center_frames'],a['predicted_token_counts'],a['mean_expected_unit_probability'])
if __name__=='__main__':main()
