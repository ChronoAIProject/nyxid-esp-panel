"""Same-wave/time float and integer policy-factor diagnosis; no threshold sweep."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ai_edge_litert.interpreter import Interpreter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_sampler_v1 import Sampler,INDEX
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from ordered_unit_event_policy_v1 import UnitEventPolicy
from ordered_generator_matcher_v1 import PATTERN,BOUNDARIES,OrderedMatcher
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');ROOT=B/'causal-generator-delay40-supervised-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(v,d):
 scale,zero=d['quantization'];return np.clip(np.rint(v/scale)+zero,-128,127).astype(np.int8)
def reference(tokens,accepted):
 recent=[];last=None;last_event=-200;events=[]
 for i,(token,ok) in enumerate(zip(tokens,accepted)):
  frame=i+1
  if frame-last_event<200 or token is None or not ok:recent=[];last=None;continue
  if recent and frame-recent[0][1]>200:recent=[];last=None
  if token==last:continue
  last=token
  if tuple(t for t,_ in recent)==PATTERN and token in BOUNDARIES:events.append(i);last_event=frame;recent=[];last=None;continue
  recent.append((token,frame));recent=recent[-len(PATTERN):]
 return events
def raw(tokens):
 m=OrderedMatcher();events=[]
 for i,t in enumerate(tokens):
  if t is None:m=OrderedMatcher()
  elif m.push(t):events.append(i)
 return events
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sampler=Sampler();mapping=sampler.data['dense_to_original_label'];cp=ROOT/'final.pt';pt=CausalGeneratorUnitModel(len(mapping)).cuda().eval();pt.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);front=ContinuousPastMFCC(torch.load(B/'matchbox-grid-aligned-v1/frontend.pt',map_location='cpu',weights_only=True)).cuda().eval();modelpath=ROOT/'int8.tflite';it=Interpreter(model_path=str(modelpath),num_threads=1);it.allocate_tensors();runner=it.get_signature_runner();ins=runner.get_input_details();outs=runner.get_output_details();assert outs['probabilities']['quantization']==(1/256,-128);records=[]
 with torch.no_grad():
  for d in sampler.descriptors():
   r=sampler.rows[d['index']];assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];clock=FrameClock();frames,ends=clock.push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None];ps=pt.encoder.initial_state(device='cuda');qs={f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(6)};fp=[];ip=[];cache_errors=[0.]*6
   for start in range(0,x.shape[2]-2,3):
    chunk=x[:,:,start:start+3];fy,ps=pt(chunk,ps);fp.append(fy.softmax(1)[0].T.cpu().numpy());result=runner(chunk=quant(chunk.cpu().numpy().transpose(0,2,1)[:,:,None,:],ins['chunk']),**qs);ip.append(result['probabilities'][0,:,0,:].astype(np.int16)+128)
    for i in range(6):
     scale,zero=outs[f'cache{i}_out']['quantization'];values=(result[f'cache{i}_out'].astype(np.float64)-zero)*scale;cache_errors[i]=max(cache_errors[i],float(np.abs(values-ps[i].cpu().numpy().transpose(0,2,1)[:,:,None,:]).max()));qs[f'cache{i}']=quant(values,ins[f'cache{i}'])
   f=np.concatenate(fp);codes=np.concatenate(ip);q=codes/256;fl=f.argmax(1);il=codes.argmax(1);ft=[mapping[int(v)]-1 if mapping[int(v)] else None for v in fl];qt=[mapping[int(v)]-1 if mapping[int(v)] else None for v in il];fc=f.max(1);ic=codes.max(1);fe=reference(ft,fc>=.5);qe=reference(qt,ic>=128);policy=UnitEventPolicy();actual=[i for i,(t,c) in enumerate(zip(qt,ic)) if policy.push(t,int(c))];assert actual==qe
   modes=dict(float_raw=raw(ft),integer_raw=raw(qt),float_span_only=reference(ft,np.ones(len(ft),bool)),integer_span_only=reference(qt,np.ones(len(qt),bool)),float_full=fe,integer_full=qe)
   records.append(dict(index=d['index'],class_id=r['class_id'],carrier=r['carrier'],source_sha256=r['sha256'],frames=len(f),argmax_disagreement_frames=int((fl!=il).sum()),float_below_half_frames=int((fc<.5).sum()),integer_below128_frames=int((ic<128).sum()),gate_disagreement_frames=int(((fc>=.5)!=(ic>=128)).sum()),mean_abs_probability_error=float(np.abs(f-q).mean()),max_abs_probability_error=float(np.abs(f-q).max()),cache_max_abs_errors=cache_errors,events_samples={k:[int(ends[i]) for i in v] for k,v in modes.items()},float_confidence_quantiles=np.quantile(fc,[0,.25,.5,.75,1]).tolist(),integer_code_quantiles=np.quantile(ic,[0,.25,.5,.75,1]).tolist()))
 summary={mode:dict(target_hits=sum(bool(r['events_samples'][mode]) for r in records if r['class_id']==1),target_count=4,negative_hits=sum(bool(r['events_samples'][mode]) for r in records if r['class_id']!=1),negative_count=20) for mode in modes}
 previous=json.load((R/'4090-causal-delay40-trained-int8-export-v1.json').open());assert all(r['events_samples']['integer_full']==old['event_samples'] for r,old in zip(records,previous['records']))
 out=dict(schema='causal-float-integer-policy-diagnostic-v1',files={str(p):sha(p) for p in [Path(__file__),cp,modelpath,INDEX]},summary=summary,records=records,actual_integer_policy_reference_agreement=True,prior_integer_replay_exact=True,qualification=False,limitations='Calibration-overlapping24training waves, identical frames/cache reset/chunk3 clock. Float uses true>=0.5 gate without fabricated UINT8 scores; independent reference validated against actual integer policy. Raw/span-only modes are explanatory factor ablations, not selected operating points. No threshold sweep or promotion. Probability/cache errors compare whole distinct recurrent trajectories, not isolated quantization kernels.')
 with (R/'4090-causal-float-integer-policy-diagnostic-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
