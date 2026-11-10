#!/usr/bin/env python3
"""Synthetic Hye Nyx voice embedding baseline; no board or recordings needed."""
from __future__ import annotations
import argparse, json, math, random, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def make_data(samples: int, speakers: int, bins: int, frames: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(speakers, (samples,), generator=g)
    centers = torch.randn(speakers, bins, generator=g) * .7
    t = torch.linspace(0, 1, frames)
    x = torch.empty(samples, bins, frames)
    for i, y in enumerate(labels):
        pitch = torch.sin(2 * math.pi * (1 + y % 5) * t)
        x[i] = centers[y].unsqueeze(1) + .18 * pitch.unsqueeze(0)
        x[i] += .12 * torch.randn(bins, frames, generator=g)
    return x, labels

class Encoder(nn.Module):
    def __init__(self, speakers: int, bins: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(bins, 64, 5, 2, 2), nn.GELU(),
                                 nn.Conv1d(64, 96, 5, 2, 2), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(96, speakers)
    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.head(z), z

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda'); p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--samples', type=int, default=10000); p.add_argument('--speakers', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261110); p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--output', type=Path, default=Path('runs/baseline')); a = p.parse_args()
    if a.samples < 5 or a.speakers < 2 or a.epochs < 1 or a.batch_size < 1:
        p.error('samples >= 5, speakers >= 2, epochs >= 1 and batch-size >= 1 required')
    if (a.output/'metrics.json').exists():
        p.error('Use a fresh output directory to preserve previous results')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device('cuda' if a.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if a.device == 'cuda' and device.type != 'cuda': raise SystemExit('CUDA requested but unavailable')
    if device.type == 'cuda': torch.backends.cuda.matmul.allow_tf32 = True
    x, y = make_data(a.samples, a.speakers, 80, 96, a.seed)
    n_val = max(1, len(y)//5); train, val = random_split(TensorDataset(x, y), [len(y)-n_val, n_val], generator=torch.Generator().manual_seed(a.seed))
    dl = DataLoader(train, a.batch_size, shuffle=True, pin_memory=device.type=='cuda')
    vd = DataLoader(val, a.batch_size, pin_memory=device.type=='cuda')
    model = Encoder(a.speakers, 80).to(device); opt = torch.optim.AdamW(model.parameters(), 2e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda'); history=[]
    for epoch in range(1, a.epochs+1):
        model.train(); total=correct=loss_sum=0
        for xb, yb in dl:
            xb,yb=xb.to(device, non_blocking=True),yb.to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type=='cuda'):
                logits,_=model(xb); loss=nn.functional.cross_entropy(logits,yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); loss_sum += loss.item()*len(yb); correct += (logits.argmax(1)==yb).sum().item(); total += len(yb)
        model.eval(); vl=vc=vn=0
        with torch.no_grad():
            for xb,yb in vd:
                logits,_=model(xb.to(device)); vl += nn.functional.cross_entropy(logits,yb.to(device)).item()*len(yb); vc += (logits.argmax(1)==yb.to(device)).sum().item(); vn += len(yb)
        row={'epoch':epoch,'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':vl/vn,'val_accuracy':vc/vn}; history.append(row)
        print(json.dumps(row), flush=True)
    confusion = torch.zeros(a.speakers, a.speakers, dtype=torch.int64)
    with torch.no_grad():
        for xb, yb in vd:
            predicted = model(xb.to(device))[0].argmax(1).cpu()
            confusion += torch.bincount(yb*a.speakers+predicted, minlength=a.speakers**2).reshape(a.speakers,a.speakers)
    a.output.mkdir(parents=True, exist_ok=True); torch.jit.script(model.cpu()).save(str(a.output/'encoder.ts'))
    exported = torch.jit.load(str(a.output/'encoder.ts')).eval()
    with torch.no_grad():
        torch.testing.assert_close(exported(x[:8])[0], model(x[:8])[0])
    final=history[-1]; (a.output/'metrics.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'export_bytes':(a.output/'encoder.ts').stat().st_size,'export_verified':True,'confusion_matrix':confusion.tolist(),'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'seed':a.seed,'config':vars(a)|{'output':str(a.output)},'history':history,'final':final}, indent=2, default=str))
    print(f"saved {a.output} | val_accuracy={final['val_accuracy']:.3f}")
if __name__ == '__main__': main()

