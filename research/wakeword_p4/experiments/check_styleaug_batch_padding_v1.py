"""Trained final-checkpoint padding/batching witness on a fixed training batch."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_prospective_sampler_v1 import Sampler
from causal_unit_full_batch_v1 import batch
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available()
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 cp=Path('.local/wakeword/causal-generator-styleaug-v1/final.pt');frontpath=Path('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt')
 s=torch.load(cp,map_location='cpu',weights_only=False);assert s['step']==5000
 sampler=Sampler(726);model=CausalGeneratorUnitModel(len(sampler.data['dense_to_original_label'])).cuda().eval();model.load_state_dict(s['model'])
 front=ContinuousPastMFCC(torch.load(frontpath,map_location='cpu',weights_only=True)).cuda().eval()
 rows=[]
 with torch.no_grad():
  x,p,mask,descriptors=batch(sampler,front);before={k:v.clone() for k,v in model.state_dict().items()};whole,_=model(x,model.encoder.initial_state(batch=24,device='cuda'))
  for i,d in enumerate(descriptors):
   n=d['frames'];single,_=model(x[i:i+1,:,:n],model.encoder.initial_state(device='cuda'));ref=whole[i:i+1,:,:n];difference=float((single-ref).abs().max());disagreement=int((single.argmax(1)!=ref.argmax(1)).sum())
   # Alter padded suffix and other records; valid prefix must remain unchanged.
   changed=x.clone();changed[i,:,n:]=1000.;changed[:i]=-1000.;changed[i+1:]=1000.
   other,_=model(changed,model.encoder.initial_state(batch=24,device='cuda'));prefix_error=float((other[i:i+1,:,:n]-ref).abs().max())
   assert torch.allclose(single,ref,atol=1e-5,rtol=1e-5)
   assert prefix_error==0
   rows.append(dict(index=d['index'],frames=n,padded_frames=x.shape[2]-n,single_vs_batch_max_abs=difference,argmax_disagreements=disagreement,other_records_and_suffix_prefix_error=prefix_error))
  assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
 out=dict(schema='styleaug-batch-padding-witness-v1',checkpoint_sha256=sha(cp),frontend_sha256=sha(frontpath),script_sha256=sha(__file__),gpu=torch.cuda.get_device_name(),torch=torch.__version__,records=rows,all_passed=True,limits='Fixed first24training records only, trained final checkpoint; no optimizer, threshold, development or acceptance access. Bounded witness, not all-input numerical or INT8 qualification.')
 dest=Path('research/wakeword_p4/results/4090-styleaug-batch-padding-witness-v1.json')
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,records=len(rows),max_abs=max(r['single_vs_batch_max_abs'] for r in rows),argmax_disagreements=sum(r['argmax_disagreements'] for r in rows))))
if __name__=='__main__':main()
