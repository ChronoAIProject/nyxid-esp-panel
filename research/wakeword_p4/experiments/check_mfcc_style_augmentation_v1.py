import hashlib,json,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_mfcc_style_augmentation_v1 import apply_style_augmentation
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.manual_seed(726);x=torch.randn(24,64,722,device='cuda',requires_grad=True);g=torch.Generator(device='cuda').manual_seed(726);a,s,b=apply_style_augmentation(x,g);g2=torch.Generator(device='cuda').manual_seed(726);c,ss,bb=apply_style_augmentation(x.detach(),g2);assert torch.equal(a.detach(),c) and torch.equal(s,ss) and torch.equal(b,bb);assert float(s.min())>=.85 and float(s.max())<=1.15 and float(b.min())>=-.15 and float(b.max())<=.15;loss=a.square().mean();loss.backward();assert torch.isfinite(x.grad).all();out=dict(schema='causal-mfcc-style-augmentation-admission-v1',gpu=torch.cuda.get_device_name(),batch=list(x.shape),deterministic_seed=726,scale_range=[.85,1.15],bias_range=[-.15,.15],repeatable=True,finite_gradients=True,optimizer_updates=0,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/wakeword_p4/training/causal_mfcc_style_augmentation_v1.py')]},policy='Training-only affine style perturbation per complete recording batch; inference graph unchanged. New training hypothesis, no recognition/qualification claim.')
 Path('research/wakeword_p4/results/4090-causal-mfcc-style-augmentation-admission-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
