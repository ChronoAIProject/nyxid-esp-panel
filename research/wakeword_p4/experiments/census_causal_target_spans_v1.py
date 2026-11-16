"""Paired training target frame-span census; no smoothing or decoder search."""
import collections,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 index=R/'4090-causal-unit-training-index-v1.json.gz';data=json.load(gzip.open(index));mapping=data['dense_to_original_label'];front=ContinuousPastMFCC(torch.load(B/'matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();runs={};files=[Path(__file__),index]
 for name,folder in [('unit','causal-generator-unit-supervised-v1'),('position','causal-generator-position-supervised-v1')]:
  cp=B/folder/'final.pt';files.append(cp);state=torch.load(cp,map_location='cpu',weights_only=False);model=CausalGeneratorUnitModel(len(mapping)).cuda().eval();model.load_state_dict(state['model']);records=[];hist=collections.Counter();conf=[]
  with torch.no_grad():
   for idx,r in enumerate(data['records']):
    if r['class_id']!=1:continue
    assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];clock=FrameClock();frames,ends=clock.push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None];y,_=model(x,model.encoder.initial_state(device='cuda'));p=y.softmax(1)[0].T.cpu().numpy();pred=p.argmax(1);confidence=p.max(1);spans=[]
    for i,label in enumerate(pred):
     token=mapping[int(label)]-1 if mapping[int(label)] else None
     if not spans or spans[-1]['token']!=token:spans.append(dict(token=token,first_frame=i,last_frame=i,frames=1,confidence_sum=float(confidence[i]),confidence_min=float(confidence[i])))
     else:s=spans[-1];s['last_frame']=i;s['frames']+=1;s['confidence_sum']+=float(confidence[i]);s['confidence_min']=min(s['confidence_min'],float(confidence[i]))
    # Bound the diagnostic region by the generator target, not by predictions.
    ids=[t['token_id'] for t in r['intervals']];start=next(i for i,t in enumerate(ids) if t==50);pattern=(50,156,47,102,16,56,156,102,53,61);assert tuple(ids[start:start+10])==pattern
    lo=r['lead']+r['intervals'][start]['start_sample'];hi=r['lead']+r['intervals'][start+9]['end_sample'];region=[]
    for s in spans:
     first=int(ends[s['first_frame']]);last=int(ends[s['last_frame']]);s['first_end_sample']=first;s['last_end_sample']=last;s['confidence_mean']=s.pop('confidence_sum')/s['frames']
     if last>=lo and first<=hi+1600:
      region.append(s)
      if s['token']==55 and last>=lo and first<=hi:hist[s['frames']]+=1;conf.append(s['confidence_mean'])
    records.append(dict(index=idx,carrier=r['carrier'],variant=r['variant'],source_sha256=r['sha256'],target_allocation_start=lo,target_allocation_end=hi,spans=region))
  runs[name]=dict(recordings=len(records),token55_span_length_frames=dict(sorted(hist.items())),token55_span_count=sum(hist.values()),token55_mean_confidence_quantiles=np.quantile(conf,[0,.25,.5,.75,1]).tolist() if conf else [],records=records)
  print(name,runs[name]['token55_span_length_frames'],flush=True)
 out=dict(schema='causal-target-frame-span-census-v1',gpu=torch.cuda.get_device_name(),files={str(p):sha(p) for p in files},runs=runs,optimizer_updates=0,limitations='All720target training recordings per model; no negatives/heldout. Region is generator allocation plus100ms right margin, not true acoustic boundary. Full predicted spans may cross region edges. Token55 anywhere within target extent need not be the specific edit from prior minimum-edit alignment. Float probabilities are not INT8 codes. No smoothing, threshold or decoder intervention.')
 with (R/'4090-causal-target-frame-span-census-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':main()
