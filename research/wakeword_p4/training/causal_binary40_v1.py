"""Untrained binary alternative on original C161 features and call phase.

Stem emits at feature indices 0,3,..., released after each three-frame call.
No batch normalization, future features or global temporal pooling.
"""
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 10)


class Binary40(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(40, 64, 3, stride=3)
        self.dw = nn.ModuleList(nn.Conv1d(64, 64, 5, dilation=d, groups=64) for d in DILATIONS)
        self.pw = nn.ModuleList(nn.Conv1d(64, 64, 1) for _ in DILATIONS)
        self.head = nn.Conv1d(64, 1, 2)

    def initial_state(self, batch=1, device=None):
        return (torch.zeros(batch, 40, 2, device=device),
                *(torch.zeros(batch, 64, 4*d, device=device) for d in DILATIONS),
                torch.zeros(batch, 64, 1, device=device))

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
    return dict(input_channels=40, channels=64, dilations=list(DILATIONS),
                feature_stride=3, stem_phase_feature_index=0,
                receptive_field_frames=3+3*(sum(4*d for d in DILATIONS)+1),
                right_context_frames=0, state_elements=2*40+64*(sum(4*d for d in DILATIONS)+1),
                conv_mac_per_call=40*64*3+5*(64*5+64*64)+64*2)
