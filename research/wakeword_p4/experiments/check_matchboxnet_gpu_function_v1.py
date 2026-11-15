"""One GPU functional call/backward on fixed pseudorandom PCM, no audio data."""
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from matchboxnet_pretrained_v1 import load,ARCHIVE_SHA

def main():
    assert torch.cuda.is_available()
    torch.manual_seed(726)
    front,model=load();front=front.cuda();model=model.cuda().eval()
    # Generate directly on CUDA; no TTS, source, calibration or final materials.
    pcm=torch.randn(2,20480,device='cuda')*.01
    features=front(pcm);logits=model(features)
    assert features.shape==(2,64,128) and logits.shape==(2,35) and torch.isfinite(logits).all()
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device='cuda'));loss.backward()
    parameters=list(model.named_parameters());assert len(parameters)>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
    report=dict(schema='matchboxnet-gpu-function-v1',device=torch.cuda.get_device_name(),torch_version=torch.__version__,weights_archive_sha256=ARCHIVE_SHA,strict_neural_tensor_load=True,all_three_original_frontend_buffers_loaded=True,learnable_parameters=sum(p.numel() for _,p in parameters),parameter_tensors=len(parameters),input_shape=list(pcm.shape),features_shape=list(features.shape),logits_shape=list(logits.shape),forward_calls=1,backward_calls=1,optimizer_updates=0,finite_logits_and_all_gradients=True,model_numerical_comparison_performed=False,real_or_synthetic_audio_opened=False,recognition_scored=False,acceptance_accessed=False,promotion_qualified=False,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py')]},limits='Basic GPU load/forward/backward feasibility only. No trained Hey Nicks head, source recognition, runtime equivalence or P4 claim. Original training commit unknown; explicit fixed128window/no variable length mask, natural-log MFCC from stored banks, symmetric conv and whole-window pooling. Not causal streaming or continuous frontend; deployment design/training lock still required.')
    out=Path('research/wakeword_p4/results/4090-matchboxnet-gpu-function-v1.json')
    with out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
