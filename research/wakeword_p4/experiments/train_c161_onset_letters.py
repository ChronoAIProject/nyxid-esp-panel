"""Fit matched small onset classifiers with and without framewise speech-recognition targets."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_c161_onset_names as onset
from check_c161_pronunciation_ctc_v1 import MODEL
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, audio, read, save, sha

STEM = "4090-c161-onset-letters"
WORK = Path(".local/wakeword/c161-onset-letters")
FIT = RESULTS/f"{STEM}-fit.json"
SOURCES = RESULTS/f"{STEM}-sources.json"
INVENTORY = RESULTS/"4090-frozen-history-teacher-metadata-math-preflight-v1.json"
CLASSES = ("Nyx","Nick","Next","Mick","other")
SEED = 631740
WIDTH = 23040
STEPS = 1000


def category(text, positive):
    if positive:
        return 0
    words = text.lower().translate(str.maketrans("", "", ",.!?")).split()
    if len(words) >= 2 and words[0] in ("hi","hey"):
        return {"nick":1,"next":2,"mick":3}.get(words[1],4)
    return 4


def sources():
    for row in read(INVENTORY)["inventory"]:
        src = row["source"]
        assert src["role"] == "training"
        label = category(src["text"],src["label"])
        yield dict(kind="piper", id=src["id"], text=src["text"], voice=f"{src['engine']}:{src['speaker_id']}",
                   label=label), audio(dict(path=src["native_path"],sha256=src["native_sha256"]))
    extra = ("Hey Next","Hi Next")
    with patch.multiple(onset, ADDRESSES=onset.ADDRESSES+extra,LABELS=onset.LABELS+(4,4)):
        for row,pcm in onset.training_sources():
            positive = row["kind"]=="kokoro" and row["label"]==0
            yield dict(row,label=category(row["text"],positive)),pcm


def teacher():
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor
    spec = read(MODEL/"source.json")["files"]
    assert spec["model.safetensors"]["sha256"] == read(RESULTS/"4090-ctc-training-audit-plan-v1.json")["expected_safetensors_sha256"]
    assert all(sha(MODEL/n)==v["sha256"] for n,v in spec.items())
    config = Wav2Vec2Config.from_pretrained(MODEL,local_files_only=True)
    config.mask_time_prob = config.mask_feature_prob = 0.
    d, info = Wav2Vec2ForCTC.from_pretrained(MODEL,config=config,local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ("missing_keys","unexpected_keys","mismatched_keys"))
    d.requires_grad_(False)
    return d.cuda().eval(), Wav2Vec2Processor.from_pretrained(MODEL,local_files_only=True), spec


@torch.inference_mode()
def encode(model,proc,waves):
    assert all(len(w)==WIDTH for w in waves)
    samples = [w.astype(np.float32)/32768 for w in waves]
    batch = proc(samples,sampling_rate=16000,return_tensors="pt").input_values.cuda()
    out = model(batch,output_hidden_states=True)
    # Six consecutive time bins of the fixed sixth transformer layer retain address order.
    latent = F.adaptive_avg_pool1d(out.hidden_states[6].transpose(1,2),6).flatten(1)
    return latent.cpu().numpy(), out.logits.cpu().numpy()


def cropped(pcm,end):
    start = max(0,end-WIDTH)
    return np.pad(pcm[start:end], (WIDTH-(end-start),0))


def collect():
    WORK.mkdir(exist_ok=True)
    model,proc,spec = teacher()
    rows, native, waves, labels = [], [], [], []
    for r,pcm in sources():
        padded = np.pad(pcm,(12000,24000))
        candidates = onset.observations(padded)
        rec = dict(**r,pcm_sha256=digest(pcm),samples=len(pcm))
        if candidates:
            start,end,patch = candidates[0]
            sample = cropped(padded,end)
            rec.update(index=len(waves),onset=start,available=end,window_sha256=digest(sample))
            native.append(patch)
            waves.append(sample)
            labels.append(r["label"])
        rows.append(rec)
    vectors, logits = [], []
    for i in range(0,len(waves),32):
        a,b = encode(model,proc,waves[i:i+32])
        vectors.append(a)
        logits.append(b.astype(np.float16))
    cache = WORK/"features.npz"
    with cache.open("xb") as f:
        np.savez_compressed(f,x=np.stack(native),labels=np.array(labels),latent=np.vstack(vectors).astype(np.float16),
                            logits=np.vstack(logits))
    save(SOURCES,dict(files={str(p):sha(p) for p in (Path(__file__),Path(onset.__file__),INVENTORY,RESULTS/onset.DAILY)},
         pretrained=spec,cache_sha256=sha(cache),records=rows,classes=CLASSES))


class Small(nn.Module):
    def __init__(self, alphabet):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,20,5,stride=(2,2),padding=2), nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20,32,3,stride=(1,2),padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,40,3,stride=(1,2),padding=1), nn.BatchNorm2d(40), nn.ReLU())
        self.temporal = nn.Sequential(nn.Conv1d(200,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),
                                      nn.Conv1d(64,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU())
        self.names = nn.Sequential(nn.AdaptiveAvgPool1d(6),nn.Flatten(),nn.Linear(384,5))
        self.letters = nn.Conv1d(64,alphabet,1)

    def forward(self,x):
        out = self.encoder(x).permute(0,1,3,2).flatten(1,2)
        seq = self.temporal(out)
        return self.names(seq), self.letters(seq).transpose(1,2)


def readout():
    return nn.Sequential(nn.LayerNorm(768*6), nn.Linear(768*6,64),nn.ReLU(),nn.Linear(64,len(CLASSES)))


def weight_file(name):
    return RESULTS/f"{STEM}-{name}.npz"


def fit():
    assert "4090" in torch.cuda.get_device_name() and read(SOURCES)["cache_sha256"] == sha(WORK/"features.npz")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with np.load(WORK/"features.npz") as z:
        labels = torch.from_numpy(z["labels"]).cuda()
        x = torch.from_numpy(z["x"]).cuda().float().unsqueeze(1)/255.
        latent = torch.from_numpy(z["latent"]).cuda().float()
        logits = torch.from_numpy(z["logits"]).cuda().float()
    soft = (logits/2).softmax(dim=2)
    blank = logits.argmax(dim=2)==0
    pools = [np.flatnonzero(labels.cpu().numpy()==i) for i in range(5)]
    reports = {}
    for mode in ("pretrained","small","distilled"):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        net = (readout() if mode=="pretrained" else Small(logits.shape[-1])).cuda()
        opt = torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.004)
        hist = []
        for step in range(STEPS):
            ix = np.concatenate([rng.choice(pool,16) for pool in pools])
            if mode == "pretrained":
                pred = net(latent[ix])
            else:
                pred, frame = net(x[ix])
            loss = F.cross_entropy(pred,labels[ix],label_smoothing=.025)
            if mode=="distilled":
                # Native feature frame centers lead to 72 outputs; pretrained valid
                # 400-sample convolution with stride 320 has 71. Align by nearest center.
                assert frame.shape[1]-1==soft.shape[1]
                frame = frame[:,:-1]
                kl = F.kl_div((frame/2).log_softmax(dim=2),soft[ix],reduction="none").sum(2)
                # Equal aggregate weight for blank and nonblank frames.
                mask = blank[ix]
                loss = loss + .5*(kl[mask].mean()+kl[~mask].mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0,99,499,STEPS-1):
                hist.append(dict(step=step+1,loss=float(loss.detach())))
        net.eval()
        with weight_file(mode).open("xb") as f:
            np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in net.state_dict().items()})
        reports[mode]=dict(history=hist,weights_sha256=sha(weight_file(mode)),parameters=sum(p.numel() for p in net.parameters()))
        print("fitted",mode,flush=True)
    save(FIT,dict(sources_sha256=sha(SOURCES),script_sha256=sha(Path(__file__)),models=reports,seed=SEED,
         steps=STEPS, train_counts=[len(p) for p in pools], gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
         policy="One final fit per mode on the same samples and class-balanced minibatches. Accept Nyx probability >= 0.5; first complete 1.32 s after VAD onset. No threshold tuning."))


if __name__ == "__main__":
    import sys
    torch.set_num_threads(1)
    {"collect":collect,"fit":fit}[sys.argv[1]]()
