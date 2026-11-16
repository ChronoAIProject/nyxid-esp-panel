"""Deterministic training-only per-record MFCC style perturbation."""
import torch
def apply_style_augmentation(x,generator,scale_range=(0.85,1.15),bias_range=(-0.15,0.15)):
 if x.ndim!=3:raise ValueError('Expected B,C,T')
 b=x.shape[0];device=x.device;scale=torch.empty((b,1,1),device=device).uniform_(*scale_range,generator=generator);bias=torch.empty((b,1,1),device=device).uniform_(*bias_range,generator=generator);return x*scale+bias,scale.squeeze().detach(),bias.squeeze().detach()
