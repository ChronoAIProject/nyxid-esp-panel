"""Fit preactivation directly on high-scoring labeled windows in the frozen recognizer."""
import sys
from unittest.mock import patch

import numpy as np
import torch

import review_c161_confuser_readout_v1 as review
import train_c161_confuser_readout_v1 as first
from fit_c161_history_gain_v1 import MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha

STEM = "4090-c161-signed-confuser"
FINAL = RESULTS / f"{STEM}-single-v1.tflite"


def main():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    ledger = read(RESULTS/f"{first.STEM}-windows-v1.json")
    it = REF_KERNEL(model_path=str(first.PARENT), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    quant = {r["index"]:r["quantization"] for r in it.get_tensor_details()}
    sx,zx = quant[62]
    sw,_ = quant[14]
    sl,zl = quant[63]
    teacher = torch.as_tensor(it.get_tensor(14).ravel(),device="cuda",dtype=torch.float32)*sw
    bias = int(it.get_tensor(13)[0])*sx*sw
    records, hidden = first.training_data()
    examples, targets, metadata = [], [], []
    for meta,h,r in zip(records,hidden,ledger["records"],strict=True):
        assert all(r[k] == value for k,value in meta.items())
        for j,start in enumerate(r["starts"]):
            if r["positive"] and j:
                break
            x = (h[start:start+5].float()-zx)*sx
            logits = ((x @ teacher+bias)/sl).round().clamp(-128-zl,127-zl)*sl
            probability = logits.sigmoid().mean().item()
            if not r["positive"] and probability < .70:
                continue
            # Negative signed targets keep the ReLU inactive on target windows.
            target = -3.0 if r["positive"] else min(8.5,max(3.0,logits.mean().item()+1.0))
            examples.append(x.mean(0))
            targets.append(target)
            metadata.append(dict(key=r["key"],voice=r["voice"],positive=r["positive"],arm=r["arm"],
                                 condition=r["condition"],start=start,teacher=probability))
    del hidden
    x = torch.stack(examples).double()
    y = torch.tensor(targets,device="cuda",dtype=torch.float64)
    pos = torch.tensor([m["positive"] for m in metadata],device="cuda",dtype=torch.bool)
    assert pos.any() and (~pos).any()
    weights = torch.where(pos, .5/pos.sum().double(), .5/(~pos).sum().double())
    mean = weights @ x
    scale = (((x-mean).square()*weights[:,None]).sum(0).sqrt()).clamp_min(.25)
    normalized = (x-mean)/scale
    intercept = weights @ y
    # Weighted ridge in the dual: preserve equal total weight for each class.
    gram = normalized @ normalized.T / x.shape[1]
    penalty = .10/(weights*len(x))
    coeff = torch.linalg.solve(gram + torch.diag(penalty), y-intercept)
    w = (coeff @ normalized)/(x.shape[1]*scale)
    offset = intercept - mean @ w
    quantized = (w/sw).round().clamp(-127,127).to(torch.int8)
    b = round(offset.item()/(sx*sw))
    with patch.object(first,"FINAL",FINAL):
        first.export(quantized.cpu().numpy(),b)
    geometry = RESULTS/f"{STEM}-windows-v1.json"
    predicted = x @ (quantized.double()*sw) + b*sx*sw
    for r, v, target in zip(metadata,predicted.tolist(),targets,strict=True):
        r.update(quantized_signed_response=v,requested_response=target)
    save(geometry,dict(records=metadata))
    save(RESULTS/f"{STEM}-training-v1.json",dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
         baseline_sha256=MODEL_SHA,parent_sha256=sha(first.PARENT),first_ledger_sha256=sha(RESULTS/f"{first.STEM}-windows-v1.json"),
         windows_sha256=sha(geometry),positive=int(pos.sum()),negative=int((~pos).sum()),ridge=.10,quant_scale=sw,
         saturated_weights=int(((w/sw).abs()>127).sum()),bias=b,nonzero=int(torch.count_nonzero(quantized)),
         reports=[dict(name="verifier",path=FINAL.name,bytes=FINAL.stat().st_size,sha256=sha(FINAL))]))


if __name__ == "__main__":
    action=sys.argv[1]
    if action=="train":
        main()
    elif action in ("transfer","audit"):
        with patch.object(review,"STEM",STEM):
            {"transfer":review.transfer,"audit":review.audit}[action]()
    else:
        import replay_c161_vowel_name_pair_v1 as replay
        with patch.object(replay,"STEM",STEM):
            for cohort in sys.argv[1:]:
                replay.run(cohort)
