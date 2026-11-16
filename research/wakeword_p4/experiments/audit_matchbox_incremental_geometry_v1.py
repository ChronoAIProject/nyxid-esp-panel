"""Static geometry, instantiate layers only; no neural forward/audio."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from matchboxnet_pretrained_v1 import MatchboxNet
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 model=MatchboxNet();blocks=[];radius=0;mac=0;history=0;residual_delay=0
 for i,b in enumerate(model.encoder.encoder):
  convs=[m for m in b.mconv if hasattr(m,'conv')];temporal=convs[0].conv;assert temporal.stride==(1,) and temporal.padding[0]==temporal.dilation[0]*(temporal.kernel_size[0]-1)//2
  r=temporal.padding[0];radius+=r
  cost=sum(m.conv.out_channels*(m.conv.in_channels//m.conv.groups)*m.conv.kernel_size[0] for m in convs)
  if b.res is not None:
   cost+=sum(m.conv.out_channels*(m.conv.in_channels//m.conv.groups)*m.conv.kernel_size[0] for branch in b.res for m in branch if hasattr(m,'conv'))
   residual_delay+=r*temporal.in_channels
  mac+=cost;history+=(2*r)*temporal.in_channels
  blocks.append(dict(block=i,input_channels=temporal.in_channels,output_channels=b.mconv[-1].num_features,kernel=temporal.kernel_size[0],dilation=temporal.dilation[0],radius_frames=r,conv_mac_per_output_frame=cost,temporal_input_cache_elements=2*r*temporal.in_channels,residual_alignment_delay_elements=r*temporal.in_channels if b.res is not None else 0))
 assert radius==54
 result=dict(schema='matchbox-incremental-geometry-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py'),Path('research/wakeword_p4/training/matchboxnet_long_window_v1.py')]},torch_version=torch.__version__,blocks=blocks,encoder_receptive_field_frames=2*radius+1,encoder_right_context_frames=radius,encoder_lookahead_ms_at10ms_hop=radius*10,conv_mac_per_frame=mac,conv_mac_per_second_at100fps=mac*100,temporal_input_cache_elements=history,residual_alignment_delay_elements=residual_delay,cache_bytes_if_all_int8_and_separate_residual=history+residual_delay,cache_bytes_if_float32_and_separate_residual=4*(history+residual_delay),head_mac_per_call=128*7,rolling_mean_ring_int8_bytes_280frames=280*128,neural_forward_calls=0,policy='Geometry only, not exported streaming graph. Symmetric stride1 encoder could delay outputs by54frames with perlayer caches; listed temporal cache is a conservative full effective history arrangement, residual alignment separately budgeted and potentially aliasable. Excludes chunk buffers, quantization/requant scales, BN/activations, pooling accumulator, frontend, arena and runtime. No proven memory lower bound or P4 peak. Continuing past original window boundaries changes original centered reflection/zero-padding and wholewindow pooling semantics, hence no behavioral inheritance. Removing lookahead changes network function; simply making symmetric layers causal cannot preserve recognition. Integer-state export/calibration remain unproven. No training/new data/qualification.')
 with (R/'4090-matchbox-incremental-geometry-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['files','blocks','policy']}))
if __name__=='__main__':main()
