"""Locked static resource estimate and one GPU functional backward; no update/audio."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-matchbox-long-window-function-plan-v1.json';OUT=R/'4090-matchbox-long-window-function-original-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 paths=[Path(__file__),Path('research/wakeword_p4/training/matchboxnet_long_window_v1.py'),Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py'),Path('.local/wakeword/compact-pretrained-feasibility-v1/matchboxnet3x1x64-v2-1.0.0rc1.nemo')]
 p=dict(schema='matchbox-long-window-function-plan-v1',files={str(x):sha(x) for x in paths},window=44800,frames=280,batch=128,seed=726,optimizer_updates=0,policy='Exactly one pseudorandom GPU MFCC+7-class forward/backward, no optimizer or actual audio/model comparison. Static conv MAC estimate and CUDA peak are host feasibility only, not P4 peak/RAM/latency or recognition. Original short adapter unchanged; no developer/final/acceptance input.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(sha(LOCK))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for x,h in p['files'].items():assert sha(x)==h
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 import torch
 from torch import nn
 from matchboxnet_long_window_v1 import load
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 front,model=load();model.decoder.decoder_layers=nn.Sequential(nn.Linear(128,7));model=model.cuda().eval();front=front.cuda().eval()
 convs=[dict(name=n,weights=m.weight.numel(),macs_per_window=280*m.weight.numel()) for n,m in model.named_modules() if isinstance(m,nn.Conv1d)]
 mac=sum(v['macs_per_window'] for v in convs)+128*7
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 wave=torch.rand(128,44800,device='cuda',dtype=torch.float32)*.2-.1
 with torch.no_grad():features=front(wave)
 logits=model(features);loss=nn.functional.cross_entropy(logits,torch.arange(128,device='cuda')%7);loss.backward();torch.cuda.synchronize()
 assert tuple(logits.shape)==(128,7) and torch.isfinite(loss) and all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
 result=dict(schema='matchbox-long-window-function-original-v1',plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,full_feature_shape=list(features.shape),trainable_parameters=sum(v.numel() for v in model.parameters()),convolutions=convs,conv_plus_dense_macs_per_window=mac,conv_plus_dense_macs_per_second_at480cadence=mac*16000/480,raw_int16_context_bytes=44800*2,mfcc_int8_input_bytes=280*64,mfcc_float32_input_bytes=280*64*4,batch128_gpu_function_seconds=time.monotonic()-start,batch128_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_functional_checks_passed=True,optimizer_updates=0,actual_audio_calls=0,p4_runtime_measured=False,recognition_scored=False,policy=p['policy'],limitations='MACs omit MFCC/FFT, BN/nonlinear/add/mean/softmax/quantization, memory copies and interpreter. CUDA batch backward peak is not single-stream P4 arena RAM or walltime. Whole finite context at480 sample cadence is a computationally costly deployable algorithm proposal, not an incremental-conv implementation.')
 with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
