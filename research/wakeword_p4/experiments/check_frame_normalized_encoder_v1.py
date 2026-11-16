import hashlib,json,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_frame_normalized_encoder_v1 import CausalFrameNormalizedEncoder
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();torch.manual_seed(726);m=CausalFrameNormalizedEncoder().cuda().eval();x=torch.randn(2,64,17,device='cuda');s=m.encoder.initial_state(2,'cuda');a,sa=m(x,s);parts=[];sp=s
 with torch.no_grad():
  for i in range(0,17,3):y,sp=m(x[:,:,i:i+3],sp);parts.append(y)
 err=float((torch.cat(parts,2)-a).abs().max());const=m.normalize(torch.ones(1,64,3,device='cuda'));assert torch.isfinite(const).all() and float(const.abs().max())==0.;assert err<1e-5
 out=dict(schema='causal-frame-normalized-encoder-function-v1',gpu=torch.cuda.get_device_name(),shape=list(x.shape),eps=m.eps,chunk3_vs_full_max_abs=err,constant_frame_zero=True,gradient_checks='pending training',optimizer_updates=0,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/wakeword_p4/training/causal_frame_normalized_encoder_v1.py')]},policy='Mean/variance over64 channels at each current frame only; no future PCM, cross-frame or cross-record statistics. New ABI requires independent INT8 audit.')
 Path('research/wakeword_p4/results/4090-causal-frame-normalized-encoder-function-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
