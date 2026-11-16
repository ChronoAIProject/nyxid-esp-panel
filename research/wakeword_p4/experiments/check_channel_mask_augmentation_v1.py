import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_channel_mask_augmentation_v1 import apply_channel_mask

def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();x=torch.randn(4,64,20,device='cuda',requires_grad=True);a=torch.Generator(device='cuda').manual_seed(726);b=torch.Generator(device='cuda').manual_seed(726);y,m=apply_channel_mask(x,a);z,n=apply_channel_mask(x,b);assert torch.equal(y,z) and torch.equal(m,n);assert m.shape==(4,64,1);assert torch.isfinite(y).all();y.sum().backward();assert torch.isfinite(x.grad).all();assert torch.equal(apply_channel_mask(x.detach(),torch.Generator(device='cuda').manual_seed(1),0.)[0],x.detach())
 p=Path('research/wakeword_p4/results/4090-channel-mask-augmentation-admission-v1.json');out=dict(schema='channel-mask-augmentation-admission-v1',gpu=torch.cuda.get_device_name(),drop_probability=.125,seed=726,deterministic=True,shape_preserved=True,finite_forward=True,finite_gradient=True,zero_probability_identity=True,inputs={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path('research/wakeword_p4/training/causal_channel_mask_augmentation_v1.py')]},limits='Admission only; no training/model/audio/development/final calls. Training-only augmentation; inference graph unchanged.')
 p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
