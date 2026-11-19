"""Learn a small readout of ordered CTC path scores at a fixed proposal clock."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import score_proposal_spellings as prior
from fit_c161_history_gain_v1 import RESULTS, REF_KERNEL, read, save, sha
from score_proposal_readout import GROUPS
from train_c161_mined_room_word import CORPUS, negative_streams

base = prior.fit.base
STEM = "4090-c161-spelling-time"
TRAIN = RESULTS / f"{STEM}-training.json"
FIT = RESULTS / f"{STEM}-fit.json"
KINDS = {
    "target": prior.TARGETS,
    "rival": prior.RIVALS,
    "greeting": ["HEY","HAY"],
    "name": ["NICKS","NIX","NYX","NIKS"],
    "reversed": [word+hey for word in ("NICKS","NIX","NYX","NIKS") for hey in ("HEY","HAY")],
}
BINS = (24,48,71)
MODES = ("global","timed","timed_synthetic")


class Features:
    def __init__(self):
        vocab = read(prior.WAV2VEC/"vocab.json")
        self.separator = vocab["|"]
        paths = [p for group in KINDS.values() for p in group]
        lengths = np.array([len(p) for p in paths])
        chars = np.zeros((len(paths),2*max(lengths)),np.int64)
        for i,p in enumerate(paths):
            chars[i,1:2*len(p):2] = [vocab[c] for c in p]
        self.symbols = torch.from_numpy(chars).cuda()
        self.skip = self.symbols[:,2:].ne(0) & self.symbols[:,2:].ne(self.symbols[:,:-2])
        self.lengths = torch.from_numpy(lengths).cuda()
        self.ends = 2*self.lengths-1

    @torch.inference_mode()
    def __call__(self, codes):
        x = torch.as_tensor(codes,device="cuda",dtype=torch.int32)
        assert x.ndim == 3 and x.shape[1:] == (BINS[-1],32)
        x = x-x.amax(2,keepdim=True)
        x[:,:,0] = torch.maximum(x[:,:,0],x[:,:,self.separator])
        n = len(x)
        state = x.new_full((n,*self.symbols.shape),-65536)
        best = x.new_full((n,len(self.ends)),-65536)
        slots = []
        for tick,data in enumerate(x.unbind(1),1):
            nxt = state.clone()
            nxt[:,:,1:] = torch.maximum(nxt[:,:,1:],state[:,:,:-1])
            nxt[:,:,2:] = torch.maximum(nxt[:,:,2:],torch.where(self.skip,state[:,:,:-2],-65536))
            nxt[:,:,1].clamp_min_(0)
            state = nxt + data[:,self.symbols]
            ends = state.gather(2,self.ends[None,:,None].expand(n,-1,1)).squeeze(2)
            best = torch.maximum(best,ends)
            if tick in BINS:
                scores = best.float()/self.lengths
                parts = scores.split([len(p) for p in KINDS.values()],dim=1)
                slots.append(torch.stack([v.amax(1) for v in parts],dim=1))
                best.fill_(-65536)
        return torch.stack(slots,dim=1).cpu().numpy()


def collect():
    vm, extract = prior.model(), Features()
    patches, _, baseline, front = prior.fit.training_windows()
    ref = read(prior.CAL)
    assert baseline == read(prior.fit.FIT)["records"]
    rows = []
    for i in range(0,len(patches),64):
        scores = extract(np.stack([prior.outputs(vm,w) for w in patches[i:i+64]]))
        for j,matrix in enumerate(scores,i):
            r = baseline[j]
            p = ref["records"][j]
            np.testing.assert_array_equal(matrix.max(0)[:2],[p["target_value"],p["rival_value"]])
            group = "target" if r["label"] == 0 else "name" if r["text"].lower().startswith(("hey ","hi ")) else "ordinary"
            rows.append(dict(key=r["key"],voice=r["voice"],view=r["kind"],group=group,
                             patch_sha256=r["patch_sha256"],features=matrix.tolist()))
    model_path = RESULTS/f"{prior.fit.parent.music.STEM}-v1.tflite"
    assert sha(model_path)==front
    proposer = REF_KERNEL(model_path=str(model_path))
    proposer.allocate_tensors()
    native = prior.fit.native_frontend()
    pending, metadata = [], []
    for s,wave in negative_streams():
        if s["corpus"] != "LibriSpeech train-clean-100":
            break
        pcm = np.pad(wave,(16000,16000))
        raw, bands, clock = base.native_inputs(native,pcm)
        trace = base.infer(proposer,bands)
        sums = np.convolve(trace.astype(np.int32),np.ones(5,np.int32),"valid")
        ids = np.flatnonzero((clock[4:]>=16000) & (clock[4:]<=16000+len(wave)))
        at = int(clock[4:][ids[np.argmax(sums[ids])]])
        ts = base.scoring.word.times(base.scoring.word.decode(trace,clock))
        eligible = [t for t in ts if 16000<=t<=16000+len(wave)]
        t = eligible[0] if eligible else at
        end = t+base.scoring.word.WAIT
        patch = np.rint(prior.fit.room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
        pending.append(prior.outputs(vm,patch))
        metadata.append(dict(key=s["key"],voice="libri/"+s["key"].split("-")[0],view="dry",group="background",
                             text=s["text"],samples=len(wave),source_sha256=s["sha256"],pcm_sha256=base.poll.digest(wave),
                             patch_sha256=base.poll.digest(patch),proposal=t,end=end,actual=bool(eligible)))
    for i in range(0,len(metadata),64):
        features = extract(np.stack(pending[i:i+64]))
        rows.extend(dict(r,features=f.tolist()) for r,f in zip(metadata[i:i+64],features,strict=True))
    save(TRAIN,dict(records=rows,split_sha256=sha(CORPUS),char_export_sha256=sha(prior.fit.EXPORT),
                    prior_training_sha256=sha(prior.CAL),script_sha256=sha(Path(__file__)),
                    groups={k:len(v) for k,v in KINDS.items()},bin_ends=BINS,
                    policy="Original training windows plus first actual proposal, else peak five-code sum, "
                    "per existing train-clean-100 recording with no target name in its transcript. Keep best "
                    "terminal CTC path value per group in frames 1-24, 25-48 and 49-71. Paths may start in "
                    "earlier bins. Target and rival global maxima must reproduce the archived training scores."))


def design(slots, mode):
    whole = slots.max(axis=1)
    return whole if mode=="global" else np.concatenate([whole,slots.reshape(len(slots),-1)],axis=1)


def apply(model, x):
    return x@np.array(model["weights"])+model["bias"]


def learn(x, rows, indices):
    data = torch.tensor(x[indices],device="cuda",dtype=torch.float64)
    source = [rows[i] for i in indices]
    keys = Counter(r["key"] for r in source)
    groups = [r["group"] for r in source]
    weights = np.array([1/keys[r["key"]] for r in source])
    totals = {"target":.5,"name":.25,"ordinary":.25}
    if "background" in groups:
        totals.update(ordinary=.125,background=.125)
    for group,mass in totals.items():
        mask = np.array(groups)==group
        assert mask.any()
        weights[mask] *= mass/weights[mask].sum()
    w = torch.from_numpy(weights).cuda()
    y = torch.tensor([g=="target" for g in groups],device="cuda",dtype=data.dtype)
    mean = (data*w[:,None]).sum(0)
    scale = ((data-mean).square()*w[:,None]).sum(0).sqrt().clamp_min(.01)
    z = (data-mean)/scale
    beta = torch.zeros(z.shape[1],device="cuda",dtype=z.dtype,requires_grad=True)
    bias = torch.zeros((),device="cuda",dtype=z.dtype,requires_grad=True)
    opt = torch.optim.LBFGS([beta,bias],max_iter=100,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9,tolerance_change=1e-12)
    def closure():
        opt.zero_grad()
        loss = (w*F.binary_cross_entropy_with_logits(z@beta+bias,y,reduction="none")).sum()+.005*beta.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    coeff = (beta/scale).detach()
    return dict(weights=coeff.tolist(),bias=(bias-coeff@mean).item())


def fit():
    assert "4090" in torch.cuda.get_device_name()
    rows = read(TRAIN)["records"]
    old = read(prior.CAL)
    slots = np.array([r["features"] for r in rows])
    y = np.array([r["group"]=="target" for r in rows])
    target_count = sum(r["label"]==0 and r["score"]>=old["threshold"] for r in old["records"])
    # Balance voices across folds without using their acoustic scores.
    voices = sorted({r["voice"] for r in rows},key=lambda v:sha256(v.encode()).digest())
    group_folds = {}
    for is_libri in (False,True):
        subset = [v for v in voices if v.startswith("libri/")==is_libri]
        group_folds.update({v:i%5 for i,v in enumerate(subset)})
    folds = np.array([group_folds[r["voice"]] for r in rows])
    reports = {}
    for mode in MODES:
        x = design(slots,mode)
        used = np.array([mode!="timed_synthetic" or r["group"]!="background" for r in rows])
        oof = np.full(len(rows),np.nan)
        for fold in range(5):
            train = np.flatnonzero(used & (folds!=fold))
            valid = np.flatnonzero(used & (folds==fold))
            model = learn(x,rows,train)
            oof[valid] = apply(model,x[valid])
        assert np.isfinite(oof[used]).all()
        threshold = float(np.sort(oof[y])[-target_count])
        final = learn(x,rows,np.flatnonzero(used))
        reports[mode] = dict(**final,threshold=threshold, oof_scores=oof[used].tolist(),
                             train_view_indices=np.flatnonzero(used).tolist(),
                             oof_accepts={g:int(np.sum((oof>=threshold) & np.array([r["group"]==g for r in rows]))) for g in sorted({r["group"] for r in rows})})
    save(FIT,dict(models=reports,voice_folds=group_folds,training_sha256=sha(TRAIN),
                  script_sha256=sha(Path(__file__)),gpu=torch.cuda.get_device_name(),
                  target_count=target_count,total_target=int(y.sum()),kind_order=list(KINDS),bin_ends=BINS,
                  policy="GPU L2 logistic regression, full-batch float64. Total weights target/name/ordinary "
                  "1/2,1/4,1/4; give half the ordinary weight to LibriSpeech when included. Give views of a "
                  "source equal total weight within its group. Fit on four of five training-voice folds. "
                  "Set each threshold to recover the old number of positive training views in out-of-fold "
                  "scores, then refit all admitted training voices. Decisions use unrounded coefficients."))


def evaluate():
    cfg = read(FIT)
    assert cfg["training_sha256"]==sha(TRAIN)
    vm, extract, native = prior.model(), Features(), prior.fit.native_frontend()
    for group in GROUPS:
        src = RESULTS/f"{prior.STEM}-{group}.json"
        letters = RESULTS/f"{prior.fit.STEM}-{group}.json"
        arms = cfg["models"]
        records = []
        for (_,pcm,c161,_,_),r,l in zip(base.sources(group),read(src)["records"],read(letters)["records"],strict=True):
            assert base.poll.digest(pcm) == r["pcm_sha256"] == l["pcm_sha256"] and c161 == r["c161"] == l["c161"]
            codes = []
            checks = []
            if r["checks"]:
                raw,_,_ = base.native_inputs(native,pcm)
            for c,d in zip(r["checks"],l["checks"],strict=True):
                assert c["end"]==d["end"]
                patch = np.rint(prior.fit.room.window(raw,c["end"]).clip(0,666)*(255./666)).astype(np.uint8)
                q = prior.outputs(vm,patch)
                assert base.poll.digest(q)==d["models"]["student"]["output_codes_sha256"]
                codes.append(q)
            accepted = {m:[] for m in arms}
            if codes:
                slots = extract(np.stack(codes)).astype(np.float64)
                np.testing.assert_array_equal(slots.max(1)[:,:2],[[c["target_value"],c["rival_value"]] for c in r["checks"]])
                scores = {m:apply(model,design(slots,m)) for m,model in arms.items()}
                for i,c in enumerate(r["checks"]):
                    end = c["end"]
                    for m,model in arms.items():
                        if scores[m][i]>=model["threshold"]:
                            accepted[m].append(end)
                    checks.append(dict(end=end,scores={m:float(v[i]) for m,v in scores.items()},features=slots[i].tolist()))
            routes = {m:base.scoring.merged(c161,v) for m,v in accepted.items()}
            records.append(dict(index=r["index"],samples=len(pcm),pcm_sha256=r["pcm_sha256"],c161=c161,
                                source=r["source"],spans=r["spans"],key=r["key"],checks=checks,routes=routes))
        save(RESULTS/f"{STEM}-{group}.json",dict(records=records,fit_sha256=sha(FIT),source_sha256=sha(src),
                                                letters_sha256=sha(letters),script_sha256=sha(Path(__file__))))
        print(group,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("collect","fit","evaluate"))
    {"collect":collect,"fit":fit,"evaluate":evaluate}[parser.parse_args().action]()
