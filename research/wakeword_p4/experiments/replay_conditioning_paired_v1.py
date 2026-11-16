"""Fixed paired final checkpoints; full-wave consumed diagnostics, not qualification."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from ordered_generator_matcher_v1 import OrderedMatcher
OUT=Path('.local/wakeword/causal-generator-unit-supervised-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['uniform','risk'],required=True);parser.add_argument('--partition',choices=['training','development'],required=True);args=parser.parse_args();root=Path('.local/wakeword/causal-conditioning-'+args.arm+'-v1')
 if args.partition=='training':
  from causal_prospective_sampler_v1 import Sampler
 else:
  from causal_prospective_development_sampler_v1 import Sampler
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 checkpoint=root/'final.pt';training=json.loads((root/'result.json').read_text());assert training['steps']==5000 and training['provenance']['arm']==args.arm;assert sha(checkpoint)==training['artifacts']['final.pt'];s=torch.load(checkpoint,map_location='cpu',weights_only=False);assert s['step']==5000
 sampler=Sampler();mapping=sampler.data['dense_to_original_label'];model=CausalGeneratorUnitModel(len(mapping)).cuda().eval();model.load_state_dict(s['model'])
 front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();results=[]
 with torch.no_grad():
  for d in [dict(index=i) for i in range(len(sampler.rows))]:
   r=sampler.rows[d['index']];assert sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];clock=FrameClock();frames,ends=clock.push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None]
   state=model.encoder.initial_state(device='cuda');parts=[]
   for start in range(0,x.shape[2],37):
    y,state=model(x[:,:,start:start+37],state);parts.append(y)
   logits=torch.cat(parts,2);full,_=model(x,model.encoder.initial_state(device='cuda'));difference=float((logits-full).abs().max());assert torch.allclose(logits,full,atol=1e-5,rtol=1e-5)
   dense=logits.argmax(1)[0].cpu().tolist();matcher=OrderedMatcher();events=[];collapsed=[];last=object()
   for i,label in enumerate(dense):
    original=mapping[label];token=original-1 if original else None
    if token!=last:collapsed.append(token);last=token
    if token is None:matcher=OrderedMatcher()
    elif matcher.push(token):events.append(int(ends[i]))
   results.append(dict(index=d['index'],class_id=r['class_id'],carrier=r['carrier'],path=r['path'],sha256=r['sha256'],frames=len(frames),events_samples=events,collapsed_tokens=collapsed,chunk37_vs_full_max_abs=difference))
 out=dict(schema='conditioning-paired-wave-diagnostic-v1',arm=args.arm,checkpoint_sha256=sha(checkpoint),script_sha256=sha(__file__),selection=args.partition+' complete-wave census; reused development, no threshold selection.',decoder='Raw float argmax exact ordered matcher; no confidence threshold, cooldown or maxspan. NOT integer event policy and NOT qualification.',records=results,target_hits=sum(bool(r['events_samples']) for r in results if r['class_id']==1),target_count=sum(r['class_id']==1 for r in results),negative_hits=sum(bool(r['events_samples']) for r in results if r['class_id']!=1),negative_count=sum(r['class_id']!=1 for r in results))
 dest=Path('research/wakeword_p4/results')/f'4090-conditioning-{args.arm}-{args.partition}-wave-diagnostic-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k!='records'}))
if __name__=='__main__':main()
