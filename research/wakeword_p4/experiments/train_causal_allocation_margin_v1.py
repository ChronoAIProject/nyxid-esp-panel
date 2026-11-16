"""Fixed-budget complete-wave CUDA generator-unit supervised baseline."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_full_batch_v1 import batch,recording_normalized_loss
from causal_conditioning_balanced_sampler_v1 import Sampler
from causal_prospective_sampler_v1 import INDEX
from causal_conditioning_risk_weights_v1 import GroupRisk
from generator_position_balanced_loss_v1 import position_weights,position_balanced_loss
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC
from contextual_generator_contrast_v1 import selected_position_labels
from contextual_allocation_margin_v1 import allocation_margin_loss
P=Path('research/wakeword_p4/causal_allocation_margin_training_plan_v1.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['allocation'],required=True);args=parser.parse_args();OUT=Path('.local/wakeword/causal-conditioning-'+args.arm+'-v1')
 lock=json.loads(P.read_text());assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes()
 for path,h in lock['files'].items():assert sha(path)==h,path
 risk=GroupRisk()
 plan=json.load(P.open());OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);assert torch.cuda.is_available();assert not (OUT/'initial.pt').exists(),'Existing run: explicit resume required'
 torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sampler=Sampler(plan['seed']);model=CausalGeneratorUnitModel(len(sampler.data['dense_to_original_label'])).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
 fp=Path('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt');front=ContinuousPastMFCC(torch.load(fp,map_location='cpu',weights_only=True)).cuda().eval()
 files=[P,INDEX,fp,Path(__file__),*[Path('research/wakeword_p4/training')/n for n in ['causal_unit_full_batch_v1.py','causal_unit_sampler_v1.py','causal_generator_unit_model_v1.py','causal_cached_encoder_v1.py','continuous_past_mfcc_v1.py','generator_overlap_targets_v1.py','generator_position_balanced_loss_v1.py','causal_conditioning_balanced_sampler_v1.py','causal_prospective_sampler_v1.py','causal_conditioning_risk_weights_v1.py','contextual_generator_contrast_v1.py','contextual_allocation_margin_v1.py']]]
 provenance=dict(arm=args.arm,files={str(p):sha(p) for p in files},plan=plan,gpu=torch.cuda.get_device_name(),torch=torch.__version__,dense_to_original_label=sampler.data['dense_to_original_label'])
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 def save(name,step):
  temp=OUT/(name+'.tmp');torch.save(dict(step=step,group_risk=risk.state(),model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance=provenance),temp);temp.replace(OUT/name)
 save('initial.pt',0);started=time.monotonic()
 with (OUT/'training.jsonl').open('x',buffering=1) as log:
  for step in range(1,plan['steps']+1):
   assert shutil.disk_usage(OUT).free>=100*1024**3
   x,p,mask,descriptors=batch(sampler,front);optimizer.zero_grad(set_to_none=True);logits,_=model(x,model.encoder.initial_state(batch=24,device='cuda'));frame_loss=recording_normalized_loss(logits,p,mask)
   positional=[]
   for i,d in enumerate(descriptors):
    r=sampler.rows[d['index']];kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {};a,ends=position_weights(r['samples'],r['lead'],r['intervals'],**kwargs);labels=[sampler.data['dense_to_original_label'].index(t['token_id']+1) for t in r['intervals']];positional.append(position_balanced_loss(logits[i,:,:len(ends)],torch.tensor(a,device='cuda'),labels))
   position_loss=torch.stack(positional).mean();contrastive=[]
   for i,d in enumerate(descriptors):
    r=sampler.rows[d['index']];sel=selected_position_labels(r['intervals'],sampler.data['dense_to_original_label'].index);n=d['frames']
    for pos,kind,label in sel:
     a,_=position_weights(r['samples'],r['lead'],r['intervals'],**(dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}));frames=torch.tensor(__import__('numpy').flatnonzero(a[pos]>0),device='cuda');frames=frames[frames<n]
     if len(frames):
      neg=sampler.data['dense_to_original_label'].index(53+1 if kind=='terminal61' else 56+1);contrastive.append(allocation_margin_loss(logits[i,:,frames],torch.full((len(frames),),label,device='cuda'),torch.full((len(frames),),neg,device='cuda'),torch.tensor(a[pos],device='cuda')[frames]))
   contrast_loss=torch.stack(contrastive).mean();loss=.4*frame_loss+.5*position_loss+.1*contrast_loss;assert torch.isfinite(loss);unweighted=float(loss.detach());group=descriptors[0]['group'];multiplier=risk.multiplier(group) if args.arm=='risk' else 1.;loss=loss*multiplier;loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
   if args.arm=='risk':risk.observe(group,unweighted)
   row=dict(step=step,group=group,multiplier=multiplier,unweighted_loss=unweighted,frame_loss=float(frame_loss.detach()),position_loss=float(position_loss.detach()),contrast_loss=float(contrast_loss.detach()),loss=float(loss.detach()),gradient_norm=float(norm),valid_frames=int(mask.sum()),elapsed_seconds=time.monotonic()-started,descriptors=descriptors);log.write(json.dumps(row)+'\n')
   if step==1 or step%50==0:print(json.dumps({k:row[k] for k in ['step','loss','gradient_norm','elapsed_seconds']}),flush=True)
   if step%250==0:save('latest.pt',step)
  save('final.pt',plan['steps'])
  result=dict(schema='causal-generator-unit-supervised-training-v1',provenance=provenance,steps=plan['steps'],elapsed_seconds=time.monotonic()-started,final_loss=row['loss'],peak_cuda_bytes=torch.cuda.max_memory_allocated(),artifacts={n:sha(OUT/n) for n in ['initial.pt','latest.pt','final.pt','training.jsonl','provenance.json']},qualification=False)
  (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
