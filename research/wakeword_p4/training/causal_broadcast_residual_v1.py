"""Untrained causal broadcast-residual adaptation, not a paper reproduction.

Retains frequency in every block. No batch/time normalization or time pooling;
ReLU replaces the reference Swish. States contain only past pooled features.
"""
import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = (1, 2, 4, 8, 10)
CHANNELS = 32
FREQUENCIES = 20


class CausalBroadcastResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, CHANNELS, (3, 5), stride=(3, 2), padding=(0, 2))
        self.spectral = nn.ModuleList(
            nn.Conv2d(CHANNELS, CHANNELS, (1, 3), padding=(0, 1), groups=CHANNELS)
            for _ in DILATIONS)
        self.temporal = nn.ModuleList(
            nn.Conv1d(CHANNELS, CHANNELS, 5, dilation=d, groups=CHANNELS)
            for d in DILATIONS)
        self.pointwise = nn.ModuleList(nn.Conv1d(CHANNELS, CHANNELS, 1) for _ in DILATIONS)
        self.head = nn.Conv1d(CHANNELS, 1, 2)

    def initial_state(self, batch=1, device=None):
        return (torch.zeros(batch, 40, 2, device=device),
                *(torch.zeros(batch, CHANNELS, 4*d, device=device) for d in DILATIONS),
                torch.zeros(batch, CHANNELS, 1, device=device))

    def forward(self, x, states):
        if x.ndim != 3 or x.shape[1] != 40 or x.shape[2] < 3 or x.shape[2] % 3:
            raise ValueError('Expected [B,40,3*N], N>=1')
        if len(states) != 7:
            raise ValueError('Expected input, five temporal, and head states')
        joined = torch.cat((states[0], x), dim=2)
        y = F.relu(self.stem(joined.transpose(1, 2).unsqueeze(1)))  # B,C,T,F
        updated = [joined[:, :, -2:]]
        for i, (spectral, temporal, pointwise, dilation) in enumerate(
                zip(self.spectral, self.temporal, self.pointwise, DILATIONS)):
            spectral_value = F.relu(spectral(y))
            pooled = spectral_value.mean(dim=3)  # B,C,T; never mean over time
            joined = torch.cat((states[i+1], pooled), dim=2)
            updated.append(joined[:, :, -4*dilation:])
            broadcast = pointwise(F.relu(temporal(joined)))
            y = F.relu(y + spectral_value + broadcast.unsqueeze(3))
        pooled = y.mean(dim=3)
        joined = torch.cat((states[-1], pooled), dim=2)
        updated.append(joined[:, :, -1:])
        return self.head(joined), tuple(updated)

    def geometry(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    channels=CHANNELS, retained_frequency_bins=FREQUENCIES,
                    dilations=list(DILATIONS), feature_stride=3,
                    receptive_field_frames=3+3*(sum(4*d for d in DILATIONS)+1),
                    state_elements=80+CHANNELS*(sum(4*d for d in DILATIONS)+1),
                    conv_mac_per_call=20*CHANNELS*3*5+5*(20*CHANNELS*3+CHANNELS*5+CHANNELS**2)+CHANNELS*2,
                    right_context_frames=0,
                    limits='Nominal convolution MAC only; excludes means/adds/activations, not measured runtime or arena.')
