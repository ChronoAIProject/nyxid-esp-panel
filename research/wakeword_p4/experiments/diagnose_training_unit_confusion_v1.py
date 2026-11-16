"""Complete training-wave allocation-weighted argmax confusion; no optimization."""
import hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_prospective_sampler_v1 import Sampler,INDEX
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from generator_position_balanced_loss_v1 import position_weights

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sampler=Sampler();mapping=sampler.data['dense_to_original_label'];models={};inputs={str(INDEX):sha(INDEX),__file__:sha(__file__)}
 for arm in ['uniform','allocation']:
  root=Path('.local/wakeword/causal-conditioning-'+arm+'-v1');cp=root/'final.pt';report=json.loads((root/'result.json').read_text());assert sha(cp)==report['artifacts']['final.pt'];s=torch.load(cp,map_location='cpu',weights_only=False);assert s['step']==5000
  model=CausalGeneratorUnitModel(len(mapping)).cuda().eval();model.load_state_dict(s['model']);models[arm]=model;inputs[str(cp)]=sha(cp)
 fp=Path('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt');inputs[str(fp)]=sha(fp);front=ContinuousPastMFCC(torch.load(fp,map_location='cpu',weights_only=True)).cuda().eval()
 groups={a:defaultdict(lambda:dict(positions=0,mass=np.zeros(len(mapping),np.float64))) for a in models}
 with torch.no_grad():
  for i,r in enumerate(sampler.rows):
   assert r['role']=='training' and sha(r['path'])==r['sha256'];pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and len(pcm)==r['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
   frames,ends=FrameClock().push(pcm);x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None]
   a,clock=position_weights(r['samples'],r['lead'],r['intervals'],**(dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}));assert np.array_equal(clock,ends)
   for arm,model in models.items():
    logits,_=model(x,model.encoder.initial_state(device='cuda'));pred=logits.argmax(1)[0].cpu().numpy()
    for pos,t in enumerate(r['intervals']):
     key=f"class{r['class_id']}:token{t['token_id']}:rir{bool(r['rir_tail'])}";g=groups[arm][key];g['positions']+=1;g['mass']+=np.bincount(pred,weights=a[pos]/a[pos].sum(),minlength=len(mapping))
   if (i+1)%500==0:print(json.dumps(dict(records=i+1,total=len(sampler.rows))),flush=True)
 result={}
 for arm,gs in groups.items():
  result[arm]={}
  for key,g in gs.items():
   assert abs(g['mass'].sum()-g['positions'])<1e-3
   result[arm][key]=dict(positions=g['positions'],prediction_mass_by_original_token={str(original-1 if original else None):float(value/g['positions']) for original,value in zip(mapping,g['mass'],strict=True) if value>0})
 out=dict(schema='training-unit-confusion-v1',inputs=inputs,records=len(sampler.rows),gpu=torch.cuda.get_device_name(),groups=result,limits='Complete training-wave per-generator-position normalized argmax mass. Not frame accuracy against human labels; mixed allocations intentionally overlap. No development/final/optimizer/threshold calls; full-forward diagnostic, not a new streaming/INT8 qualification.')
 p=Path('research/wakeword_p4/results/4090-training-unit-confusion-v1.json')
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(completed=True,records=len(sampler.rows))))
if __name__=='__main__':main()
