"""Single 128-channel capacity alternative on original Binary40 geometry.

Stem emits at feature indices 0,3,..., released after each three-frame call.
No batch normalization, future features or global temporal pooling.
"""
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 10)


class Binary40Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(40, 128, 3, stride=3)
        self.dw = nn.ModuleList(nn.Conv1d(128, 128, 5, dilation=d, groups=128) for d in DILATIONS)
        self.pw = nn.ModuleList(nn.Conv1d(128, 128, 1) for _ in DILATIONS)
        self.head = nn.Conv1d(128, 1, 2)

    def initial_state(self, batch=1, device=None):
        return (torch.zeros(batch, 40, 2, device=device),
                *(torch.zeros(batch, 128, 4*d, device=device) for d in DILATIONS),
                torch.zeros(batch, 128, 1, device=device))

    def forward(self, x, states):
        if x.ndim != 3 or x.shape[1] != 40 or x.shape[2] < 3 or x.shape[2] % 3:
            raise ValueError('Expected [B,40,3*N], N>=1; host retains incomplete calls')
        if len(states) != 7:
            raise ValueError('Expected stem, five block and head caches')
        joined = torch.cat((states[0], x), dim=2)
        y = F.relu(self.stem(joined))
        updated = [joined[:, :, -2:]]
        for i, (dw, pw, dilation) in enumerate(zip(self.dw, self.pw, DILATIONS)):
            joined = torch.cat((states[i+1], y), dim=2)
            updated.append(joined[:, :, -4*dilation:])
            y = F.relu(pw(F.relu(dw(joined))) + y)
        joined = torch.cat((states[-1], y), dim=2)
        updated.append(joined[:, :, -1:])
        return self.head(joined), tuple(updated)


def geometry():
    return dict(input_channels=40, channels=128, dilations=list(DILATIONS),
                feature_stride=3, stem_phase_feature_index=0,
                receptive_field_frames=3+3*(sum(4*d for d in DILATIONS)+1),
                right_context_frames=0, state_elements=2*40+128*(sum(4*d for d in DILATIONS)+1),
                conv_mac_per_call=40*128*3+5*(128*5+128*128)+128*2)


def widen_from(original, seed=727):
    """Duplicate channels; zero-sum outgoing splits break gradient symmetry.

    Initial function preservation is numerical and must be measured, not assumed.
    Noise magnitude is fixed relative to each original layer RMS, no sweep.
    """
    wide=Binary40Wide().to(next(original.parameters()).device)
    generator=torch.Generator(device=next(original.parameters()).device).manual_seed(seed)
    def split(w):
        base=w.repeat(2,2,1)/2
        noise=torch.randn((base.shape[0],w.shape[1],w.shape[2]),device=w.device,generator=generator)*w.square().mean().sqrt()*0.01
        return base+torch.cat((noise,-noise),dim=1)
    with torch.no_grad():
        wide.stem.weight.copy_(original.stem.weight.repeat(2,1,1));wide.stem.bias.copy_(original.stem.bias.repeat(2))
        for a,b in zip(wide.dw,original.dw):
            a.weight.copy_(b.weight.repeat(2,1,1));a.bias.copy_(b.bias.repeat(2))
        for a,b in zip(wide.pw,original.pw):
            a.weight.copy_(split(b.weight));a.bias.copy_(b.bias.repeat(2))
        w=original.head.weight;noise=torch.randn(w.shape,device=w.device,generator=generator)*w.square().mean().sqrt()*0.01
        wide.head.weight.copy_(torch.cat((w/2+noise,w/2-noise),dim=1));wide.head.bias.copy_(original.head.bias)
    return wide
