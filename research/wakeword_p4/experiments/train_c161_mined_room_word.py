"""Match random and high-scoring training-only speech windows for fine-tuning the compact room classifier."""
import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from fit_c161_history_gain_v1 import DAILY, RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_room_word import STEM as PARENT, room_render
from train_c161_word_window_v1 import WordWindow, WIDTH, bands, training_data

PREFIX = "4090-c161-room-word-negatives"
CORPUS = RESULTS/"4090-bounded-int8-continuous-training-background-v1.json"
WORK = Path(".local/wakeword/c161-room-word")
SEED = 62417
STEPS = 420


def parent_model():
    file = RESULTS/f"{PARENT}-weights-v1.npz"
    assert sha(file)==read(RESULTS/f"{PARENT}-training-v1.json")["weights_sha256"]
    net = WordWindow().cuda()
    with np.load(file) as weights:
        net.load_state_dict({k:torch.from_numpy(weights[k]) for k in weights.files})
    return net


def negative_streams():
    transcripts = {}
    for r in read(CORPUS)["sources"]:
        p=Path(r["path"])
        assert "train-clean-100" in p.parts and sha(p)==r["sha256"]
        if p.parent not in transcripts:
            transcript_file, = p.parent.glob("*.trans.txt")
            transcripts[p.parent] = dict(line.split(" ",1) for line in transcript_file.read_text().splitlines())
        text=transcripts[p.parent][p.stem]
        if re.search(r"\b(NICKS?|NYX|NIX|NIKS)\b",text,re.I):
            continue
        pcm,rate=sf.read(p,dtype="int16")
        assert rate==16000 and hashlib.sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
        yield dict(key=p.stem,corpus="LibriSpeech train-clean-100",text=text,sha256=r["sha256"]),pcm
    augment=room_render()
    for r in read(RESULTS/DAILY)["admitted"]:
        req=r["request"]
        assert req["role"]=="training-only" and req["requested_label"]==0
        original=load_wav(r["wav"])
        pcm=np.r_[np.zeros(8000,np.int16),original,np.zeros(8000,np.int16)]
        for view in ("clean","augmented"):
            key=f'daily_{r["index"]}_{view}'
            wave=pcm if view=="clean" else augment(dict(key=key),pcm,8000)[0]
            yield dict(key=key,corpus="Chatterbox daily training",text=req["text"],
                       sha256=r["wav"]["sha256"]),wave


def mine():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    net=parent_model().eval()
    rng=np.random.default_rng(SEED)
    banks={"random":[], "hard":[]}
    rows=[]
    with torch.inference_mode():
        for src,pcm in negative_streams():
            values,clock=bands(pcm)
            windows=np.lib.stride_tricks.sliding_window_view(values,WIDTH,axis=0)
            ends=clock[WIDTH-1:]
            # Select complete five-tick spans ending inside the original utterance.
            eligible=np.flatnonzero((ends[:-4]>0)&(ends[4:]<=len(pcm)))
            if not len(eligible):
                continue
            x=torch.from_numpy(windows.copy()).cuda().float()/255
            probs=net(x).sigmoid()[:,0].cpu().numpy()
            scores=np.convolve(probs,np.ones(5)/5,mode="valid")
            row=dict(**src,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),samples=len(pcm))
            for name,order in (("random",rng.permutation(eligible)),("hard",eligible[np.argsort(-scores[eligible],kind="stable")])):
                i=int(order[0])
                banks[name].append(windows[i:i+5].copy())
                row[name]=dict(end=int(ends[i+4]),five_average=float(scores[i]))
            rows.append(row)
    cache=WORK/"negative-banks.npz"
    with cache.open("xb") as f:
        np.savez_compressed(f,**{k:np.stack(v) for k,v in banks.items()})
    save(RESULTS/f"{PREFIX}-mining-v1.json",dict(corpus_sha256=sha(CORPUS),cache_sha256=sha(cache),seed=SEED,
                                                parent_weights=sha(RESULTS/f"{PARENT}-weights-v1.npz"),records=rows))


def train():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    x, y, near, data=training_data(room_render())
    old=read(RESULTS/f"{PARENT}-training-v1.json")
    assert [r["stream_sha256"] for r in data]==[r["stream_sha256"] for r in old["training"]]
    cache=WORK/"negative-banks.npz"
    mining=RESULTS/f"{PREFIX}-mining-v1.json"
    assert sha(cache)==read(mining)["cache_sha256"]
    positive,negative,confuser=[np.flatnonzero(v) for v in (y==1,y==0,(y==0)&near)]
    with np.load(cache) as bank:
        for mode in ("random","hard"):
            torch.manual_seed(SEED)
            rng=np.random.default_rng(SEED)
            model=parent_model()
            opt=torch.optim.AdamW(model.parameters(),lr=.00025,weight_decay=.001)
            added=torch.from_numpy(bank[mode]).cuda().float()/255
            history=[]
            for step in range(STEPS):
                idx=np.r_[rng.choice(positive,64),rng.choice(negative,96),rng.choice(confuser,48)]
                speech=added[rng.choice(len(added),32)].reshape(-1,20,WIDTH)
                logits=model(torch.cat((x[idx],speech)))[:,0]
                pos,neg,extra=torch.split(logits,[64,len(idx)-64,len(speech)])
                loss=F.softplus(4.5-pos).mean()+F.softplus(neg+2).mean()
                loss+=F.softplus(neg.topk(24).values+2).mean()+F.softplus(extra+2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step in (0,99,STEPS-1):
                    history.append(dict(step=step+1,loss=loss.item()))
            stem=f"{PREFIX}-{mode}"
            weights=RESULTS/f"{stem}-weights-v1.npz"
            with weights.open("xb") as f:
                np.savez_compressed(f,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            calibration=RESULTS/f"{stem}-calibration-v1.npz"
            selected=np.r_[rng.choice(positive,96,replace=False),rng.choice(negative,128,replace=False)]
            extra=added[rng.choice(len(added),32,replace=False),2]
            inputs=torch.cat((x[selected],extra))
            with torch.no_grad():
                prob=model(inputs).sigmoid()[:,0].cpu().numpy()
            with calibration.open("xb") as f:
                np.savez_compressed(f,inputs=(inputs.cpu().numpy()*255).round().astype(np.uint8),probabilities=prob)
            save(RESULTS/f"{stem}-training-v1.json",dict(mode=mode,seed=SEED,steps=STEPS,gpu=torch.cuda.get_device_name(),
                 parent_training_sha256=sha(RESULTS/f"{PARENT}-training-v1.json"),mining_sha256=sha(mining),
                 history=history,weights_sha256=sha(weights),calibration_sha256=sha(calibration)))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="mine":
        mine()
    elif action=="train":
        train()
    else:
        for mode in ("random","hard"):
            stem=f"{PREFIX}-{mode}"
            if action=="export":
                import export_c161_word_window_v1 as runner
            elif action=="replay":
                import replay_c161_word_window_v1 as runner
            else:
                raise ValueError(action)
            runner.STEM=stem
            runner.main()
