"""Discriminatively fit second-word PTM mixture weights against marked training-only name prefixes."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_q8_keyword_map import FILES, FRAME, GAUSSIAN_SHAPE, MAP, ROOT as MAP_ROOT, SOURCES, WEIGHTS as MAP_WEIGHTS
from train_c161_word_window_v1 import sources

STEM = "4090-c161-q8-name-contrast"
ROOT = Path(".local/wakeword/c161-q8-name-contrast")
WEIGHTS = RESULTS/f"{STEM}.sendump"
STEPS = 100
LOG_UNIT = 1024*np.log(1.0001)


def align():
    index = {r["key"]:r for r in read(SOURCES)["training"]}
    table = {}
    for line in MAP.read_text().splitlines():
        fields = line.split()
        if len(fields) == 10 and fields[0] in ("N","IH","K","S") and fields[3] != "-":
            table[int(fields[5])] = list(map(int,fields[6:9]))
    assert len(table) == 4
    owners, states, phones, features, labels = [], [], [], [], []
    seen, rows = {}, []
    for src, pcm in sources():
        prior = index.pop(src["key"])
        assert prior["pcm_sha256"] == digest(pcm) and prior["positive"] == src["positive"]
        if src["end"] is None or (not src["positive"] and not src["text"].lower().startswith("hey ")):
            continue
        assert 0 < src["end"] <= len(pcm)
        prefix = pcm[:src["end"]]
        key = digest(prefix)
        row = dict(key=src["key"], positive=src["positive"], text=src["text"], end=src["end"], prefix_sha256=key)
        rows.append(row)
        if key in seen:
            assert seen[key] == src["positive"]
            row["status"] = "duplicate"
            continue
        seen[key] = src["positive"]
        trial = ROOT/"prefix.pcm"
        np.pad(prefix,(8000,4800)).astype("<i2").tofile(trial)
        result = subprocess.run([str(MAP_ROOT/"align"),str(FILES/"am"),str(FILES/"two.dict"),str(trial)],capture_output=True)
        if result.returncode == 2:
            row["status"] = "alignment failed"
            continue
        result.check_returncode()
        frames = np.frombuffer(result.stdout,FRAME)
        frames = frames[np.isin(frames["tags"][:,1],list(table))]
        if set(frames["tags"][:,1]) != set(table) or np.any(frames["tags"][:,2] < 0) or np.any(frames["tags"][:,2] > 2):
            row["status"] = "incomplete word"
            continue
        tags = frames["tags"]
        states.extend(table[phone][position] for _,phone,position,_ in tags)
        phones.extend(tags[:,1])
        features.append(frames["feature"])
        owners.extend([len(labels)]*len(frames))
        labels.append(src["positive"])
        row.update(status="used", frames=len(frames), alignment_sha256=hashlib.sha256(result.stdout).hexdigest())
    assert not index
    return (np.concatenate(features), np.array(owners), np.array(states), np.array(phones), np.array(labels), rows)


def main():
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    plan = RESULTS/f"{STEM}-plan.json"
    parent = RESULTS/"4090-c161-q8-keyword-map-fit.json"
    assert sha(MAP_WEIGHTS)==read(parent)["adapted_sendump_sha256"] and sha(MAP_ROOT/"align")==read(parent)["helper_sha256"]
    recipe = dict(parent_fit_sha256=sha(parent),source_sha256=sha(SOURCES),script_sha256=sha(Path(__file__)),
         steps=STEPS, lr=.025, kl_weight=1., temperature=4.,feature="native 3x13, forced keyword alignment",
         slots="only N IH K S context-state mixture distributions; source MAP hey and original CI phone loop",
         objective="class-balanced log loss of mean aligned keyword-state log likelihood minus same-phone CI-state log likelihood, plus KL from MAP weights",
         cases="all positive and Hey-prefixed negative training sources with recorded prefix bounds, deduplicated by prefix PCM SHA")
    if plan.exists():
        assert read(plan) == recipe
    else:
        save(plan,recipe)
    x, owner, state, phone, is_positive, rows = align()
    values = torch.as_tensor(x,device="cuda",dtype=torch.float32)
    parent_data = MAP_WEIGHTS.read_bytes()
    dims = (3,128,144)
    native = np.frombuffer(parent_data[-np.prod(dims):],np.uint8).reshape(dims).transpose(2,0,1).copy()
    logprior = (-torch.tensor(native,device="cuda").float()*LOG_UNIT).log_softmax(-1)
    ids = np.unique(state)
    state_ix = torch.tensor(np.searchsorted(ids,state),device="cuda")
    center = logprior[ids]
    logits = torch.nn.Parameter(center.clone())
    means, var = [torch.as_tensor(np.frombuffer((FILES/"am"/f).read_bytes()[72:-4],"<f4").copy().reshape(GAUSSIAN_SHAPE),device="cuda") for f in ("means","variances")]
    var = var.clamp_min(.0001)
    log_acoustic = torch.empty((len(x),3,128),device="cuda")
    for ph in np.unique(phone):
        indices = np.flatnonzero(phone==ph)
        for at in range(0,len(indices),256):
            take = indices[at:at+256]
            delta = values[take,:,None,:]-means[ph]
            log_acoustic[take] = -.5*(delta.square()/var[ph]+var[ph].log()).sum(-1)
    # Each set of three context states in the compact definition is ordered by emitting state.
    ci = phone*3+(state-126)%3
    filler = torch.logsumexp(log_acoustic+logprior[ci],-1)
    log_acoustic -= filler[:,:,None]
    owner = torch.as_tensor(owner,device="cuda")
    n = torch.bincount(owner)
    positive = torch.tensor(is_positive,device="cuda")
    opt = torch.optim.Adam([logits],lr=.025)
    history = []
    for step in range(STEPS):
        current = logits.log_softmax(-1)
        frames = torch.logsumexp(log_acoustic+current[state_ix],-1).sum(-1)
        score = logits.new_zeros(len(n)).index_add_(0,owner,frames)/n
        loss = .5*(F.softplus(-4*score[positive]).mean()+F.softplus(4*score[~positive]).mean())
        loss = loss + (center.exp()*(center-current)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,24,49,99):
            history.append(dict(step=step+1,loss=float(loss.detach()),mean_positive=float(score[positive].mean().detach()),
                                mean_negative=float(score[~positive].mean().detach())))
    codes = torch.ceil(-logits.detach().log_softmax(-1)/LOG_UNIT).clamp(0,159).byte().cpu().numpy()
    updated = native.copy()
    updated[ids] = codes
    assert len(ids)==12 and np.array_equal(updated[:126],native[:126]) and np.isfinite(history[-1]["loss"])
    with WEIGHTS.open("xb") as f:
        f.write(parent_data[:-np.prod(dims)]+updated.transpose(1,2,0).tobytes())
    folder = ROOT/"am"
    folder.mkdir(exist_ok=True)
    for f in (FILES/"am").iterdir():
        shutil.copyfile(WEIGHTS if f.name=="sendump" else f,folder/f.name)
    save(RESULTS/f"{STEM}-fit.json",dict(plan_sha256=sha(plan),adapted_sendump_sha256=sha(WEIGHTS),
         parent_sha256=sha(MAP_WEIGHTS),device=torch.cuda.get_device_name(),rows=rows,positive=int(sum(is_positive)),
         negative=int(sum(~is_positive)),frames=len(x),history=history,changed=int(np.count_nonzero(updated!=native))))
    print("fit",len(is_positive),len(x),history,flush=True)


if __name__ == "__main__":
    main()
