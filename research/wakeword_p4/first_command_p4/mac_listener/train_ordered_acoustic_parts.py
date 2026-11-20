"""Learn three ordered acoustic parts on training-only speech and calibrate integer streaming scores."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_proposal_letters as frontend
import train_c161_word_window_v1 as words
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha

STEM = "4090-c161-ordered-acoustic-parts"
FIT = RESULTS / (STEM + "-fit.json")
WORK = Path(".local/wakeword/c161-ordered-acoustic-parts")
INVENTORY = RESULTS / "4090-c161-music-proposal-training-v1.json"
PLAN = RESULTS.parent / "experiments/c161_ordered_acoustic_parts_protocol.md"
GAPS = {"fixed": ((5,5),(6,6)), "variable": ((2,8),(3,9))}
SEED = 80713


def collect():
    inventory = {r["key"]:r for r in read(INVENTORY)["records"]}
    records, features, masks, seen = [], [], [], {}
    native = frontend.native_frontend()
    for source, pcm in words.sources():
        prior = inventory.pop(source["key"])
        digest = sha256(pcm.tobytes()).hexdigest()
        assert prior["pcm_sha256"] == digest and prior["positive"] == source["positive"]
        if digest in seen:
            other = seen[digest]
            assert other["positive"] == source["positive"] and other["voice"] == source["voice"]
            other["aliases"].append(source["key"])
            continue
        stream = np.pad(pcm,(24000,16000))
        _, bands, clock = frontend.base.native_inputs(native, stream)
        # Keep causal history for scoring even the first physical live frame.
        keep = clock >= -32*480
        bands, clock = bands[keep], clock[keep]
        mask = clock > 0
        if source["positive"]:
            end = 24000 + source["end"]
            mask &= (clock <= end) & (clock >= end-4320)
        assert mask.sum() >= 2
        row = dict(**source, aliases=[], pcm_sha256=digest, frames=len(bands))
        seen[digest] = row
        records.append(row)
        features.append(torch.from_numpy(bands))
        masks.append(torch.from_numpy(mask))
    assert not inventory
    x = nn.utils.rnn.pad_sequence(features,batch_first=True).cuda().transpose(1,2).contiguous().float()/255
    valid = nn.utils.rnn.pad_sequence(masks,batch_first=True).cuda()
    return records, features, x, valid


class Parts(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Conv1d(20,16,5)
        self.parts = nn.Conv1d(16,3,1)

    def forward(self, x, gaps):
        h = self.hidden(F.pad(x,(4,0))).clamp(0,1)
        a,b,c = self.parts(h).unbind(1)
        total = a
        for local,(minimum,maximum) in zip((b,c),gaps,strict=True):
            # The latest eligible predecessor ends 'minimum' frames before this part.
            shifted = F.pad(total, (maximum,0), value=-1e5)[:,:-minimum]
            total = local + F.max_pool1d(shifted.unsqueeze(1),maximum-minimum+1,stride=1).squeeze(1)
        return torch.minimum(total, F.pad(total[:,:-1],(1,0),value=-1e5))


def quantize(net):
    values = {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
    def weights(data, scale):
        return np.rint(data*scale).clip(-127,127).astype(np.int8)
    return dict(w1=weights(values["hidden.weight"].transpose(0,2,1).reshape(16,100),64),
                b1=np.rint(values["hidden.bias"]*255*64).astype(np.int32),
                w2=weights(values["parts.weight"][:,:,0],32),
                b2=np.rint(values["parts.bias"]*255*32).astype(np.int32))


def integer_scores(bands, weights, gaps):
    # Work in bounded batches so long recordings do not allocate a full expanded stack.
    parts = np.empty((len(bands),3), np.int32)
    padded = np.pad(bands, ((4,0),(0,0)))
    for start in range(0,len(bands),2048):
        end = min(len(bands),start+2048)
        windows = np.lib.stride_tricks.sliding_window_view(padded[start:end+4],5,axis=0)
        patches = windows.transpose(0,2,1).reshape(end-start,100).astype(np.int32)
        accum = patches @ weights["w1"].astype(np.int32).T + weights["b1"]
        hidden = np.clip((accum+32)//64,0,255)
        parts[start:end] = hidden @ weights["w2"].astype(np.int32).T + weights["b2"]
    total = parts[:,0]
    for local, (minimum,maximum) in zip(parts.T[1:],gaps,strict=True):
        past = np.pad(total,(maximum,0),constant_values=-100000000)[:-minimum]
        total = local + np.lib.stride_tricks.sliding_window_view(past,maximum-minimum+1).max(axis=1)
    return np.minimum(total,np.r_[-100000000,total[:-1]]).astype(np.int32)


def main():
    device = torch.cuda.get_device_name()
    assert "4090" in device and sha(MODEL) == MODEL_SHA
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    records, features, x, valid = collect()
    voices = sorted({r["voice"] for r in records},key=lambda s:sha256(s.encode()).digest())
    held = voices[::5]
    for r in records:
        r["split"] = "calibration" if r["voice"] in held else "fit"
    pools = [np.array([i for i,r in enumerate(records) if r["split"]=="fit" and r["positive"]==v]) for v in (True,False)]
    models = {}
    WORK.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        for name,gaps in GAPS.items():
            torch.manual_seed(SEED)
            rng = np.random.default_rng(SEED)
            net = Parts().cuda()
            optimizer = torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.001)
            history = []
            for step in range(700):
                ids = np.r_[rng.choice(pools[0],32),rng.choice(pools[1],64)]
                logits = net(x[ids],gaps)
                peak = logits.masked_fill(~valid[ids],-1e5).amax(1)
                loss = F.softplus(-peak[:32]).mean() + F.softplus(peak[32:]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step in (0,99,349,699):
                    history.append([step+1,float(loss.detach())])
            params = quantize(net)
            filename = STEM+"-"+name+".npz"
            np.savez_compressed(temp/filename,**params)
            for i,r in enumerate(records):
                scores = integer_scores(features[i].numpy(),params,gaps)
                r.setdefault("peak",{})[name] = int(scores[valid[i,:len(scores)].cpu().numpy()].max())
            negatives = [r for r in records if r["split"]=="calibration" and not r["positive"]]
            cut = max(r["peak"][name] for r in negatives)+1
            models[name] = dict(file=filename,sha256=sha(temp/filename),cutoff=cut,gaps=gaps,loss=history,
                                parameter_bytes=sum(a.nbytes for a in params.values()))
            print(name, models[name], flush=True)
        save(temp/FIT.name,dict(records=records,models=models,calibration_voices=held,gpu=device,torch=torch.__version__,
             seed=SEED,steps=700,c161=MODEL_SHA,inventory_entries=918,
             sources={p.name:sha(p) for p in (PLAN,INVENTORY,Path(__file__),Path(words.__file__),
                                               frontend.base.BUILD/"frontend.so")}))
        for p in temp.iterdir():
            p.replace(RESULTS/p.name)


if __name__ == "__main__":
    main()
