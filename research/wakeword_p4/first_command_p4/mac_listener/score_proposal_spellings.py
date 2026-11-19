"""Score ordered CTC spelling paths against competing calls in actual quantized student outputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

import train_proposal_letters as fit
from check_c161_address_grammar import NAMES
from check_c161_pronunciation_ctc_v1 import MODEL as WAV2VEC
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_proposal_readout import GROUPS

STEM = "4090-c161-proposal-spellings"
CAL = RESULTS/f"{STEM}-training.json"
TARGETS = [lead+name for lead in ("HEY","HAY") for name in ("NICKS","NIX","NYX","NIKS")]
RIVALS = [lead+word.upper() for lead in ("HEY","HAY") for word in (*NAMES,"willow","mike") if word != "nicks"]
RIVALS += ["NICEMIX","PLEASEMIX","ITHINK"]


class Matcher:
    def __init__(self):
        self.vocab = read(WAV2VEC/"vocab.json")
        self.paths = TARGETS+RIVALS
        self.ends = torch.tensor([len(p)*2-1 for p in self.paths],device="cuda")
        self.denom = torch.tensor([len(p) for p in self.paths],device="cuda")
        chars = np.zeros((len(self.paths),int(self.ends.max())+1),np.int64)
        for i,p in enumerate(self.paths):
            chars[i,1:len(p)*2:2] = [self.vocab[c] for c in p]
        self.chars = torch.from_numpy(chars).cuda()
        self.skip = self.chars[:,2:].ne(self.chars[:,:-2]) & self.chars[:,2:].ne(0)

    @torch.inference_mode()
    def scores(self, outputs):
        # Frame-wise unconstrained best is zero. All node costs are signed integer deficits.
        x = torch.as_tensor(outputs,device="cuda",dtype=torch.int32)
        x -= x.amax(2,keepdim=True)
        x[:,:,0] = torch.maximum(x[:,:,0],x[:,:,self.vocab["|"]])
        n = len(x)
        bad = -65536
        state = x.new_full((n,*self.chars.shape),bad)
        best = x.new_full((n,len(self.paths)),bad)
        for data in x.unbind(1):
            nxt = state.clone()
            nxt[:,:,1:] = torch.maximum(nxt[:,:,1:],state[:,:,:-1])
            nxt[:,:,2:] = torch.maximum(nxt[:,:,2:],torch.where(self.skip,state[:,:,:-2],bad))
            nxt[:,:,1].clamp_min_(0)  # begin this spelling at any frame
            state = nxt + data[:,self.chars]
            best = torch.maximum(best,state.gather(2,self.ends[None,:,None].expand(n,-1,1)).squeeze(2))
        # Divide by number of characters; report absolute fit as well as the margin over other names.
        out = best.float()/self.denom
        target, ti = out[:,:len(TARGETS)].max(1)
        rival, ri = out[:,len(TARGETS):].max(1)
        rows = []
        for a,b,i,j in zip(target.tolist(),rival.tolist(),ti.tolist(),ri.tolist(),strict=True):
            rows.append(dict(target=TARGETS[i],rival=RIVALS[j],target_value=a,rival_value=b,score=2*a-b))
        return rows


def outputs(vm, patch):
    vm.vm.set_tensor(vm.inp["index"],vm.table[patch][None,:,:,None])
    vm.vm.invoke()
    return vm.vm.get_tensor(vm.out["index"])[0,:-1,0,:]


def model():
    spec = read(fit.EXPORT)["models"]["student"]
    path = RESULTS/spec["file"]
    assert sha(path) == spec["sha256"]
    return fit.CharacterModel(path)


def calibrate():
    patches,_,rows,front = fit.training_windows()
    old = read(fit.FIT)
    assert rows == old["records"] and front == old["front_sha256"]
    vm, matcher = model(), Matcher()
    scores = []
    for start in range(0,len(patches),96):
        q = np.stack([outputs(vm,x) for x in patches[start:start+96]])
        scores += matcher.scores(q)
    actual = np.array([r["label"] == 0 for r in rows])
    names = np.array([r["text"].lower().startswith(("hey ","hi ")) for r in rows])
    groups = [actual, ~actual & names, ~actual & ~names]
    values = np.array([s["score"] for s in scores])
    # Same aggregate calibration weighting as the existing PocketSphinx training-source rule.
    thresholds = np.r_[np.unique(values),np.nextafter(values.max(),np.inf)]
    def rank(t):
        accepted = values >= t
        positive, named, other = [np.mean((accepted==actual)[idx]) for idx in groups]
        return (.5*positive+.25*named+.25*other,other,positive,t)
    threshold = float(max(thresholds,key=rank))
    labels = [dict(key=r["key"],kind=r["kind"],label=r["label"],text=r["text"],**s) for r,s in zip(rows,scores,strict=True)]
    save(CAL,dict(threshold=threshold,threshold_rank=rank(threshold),group_counts=[int(x.sum()) for x in groups],
         records=labels,fit_sha256=sha(fit.FIT),export_sha256=sha(fit.EXPORT),script_sha256=sha(Path(__file__)),
         target_spellings=TARGETS,competing_spellings=RIVALS,vocab_sha256=sha(WAV2VEC/"vocab.json"),
         policy="Free start/end Viterbi CTC, per-frame deficit from best signed integer logit; "
         "collapse separator into blank, skip only across distinct consecutive letters. Best score for a "
         "spelling is divided by its length. Statistic = target fit + (target fit - rival fit). "
         "Choose threshold on the 1525 previously used training views: correct positives half, correct "
         "named negatives and ordinary negatives one quarter each; ties prefer ordinary rejection, then "
         "positive recall, then stricter threshold. Development audio excluded."))


def evaluate():
    cal = read(CAL)
    assert cal["export_sha256"]==sha(fit.EXPORT) and cal["script_sha256"]==sha(Path(__file__))
    vm, matcher = model(), Matcher()
    native = fit.native_frontend()
    for group in GROUPS:
        letters_file = RESULTS/f"{fit.STEM}-{group}.json"
        reference_file = RESULTS/f"{fit.reference.STEM}-{group}.json"
        records = []
        for (_,pcm,c161,_,_),a,r in zip(fit.base.sources(group),read(letters_file)["records"],read(reference_file)["records"],strict=True):
            assert fit.base.poll.digest(pcm)==a["pcm_sha256"]==r["pcm_sha256"] and c161==a["c161"]==r["c161"]
            checks = [c for c in r["checks"] if "verifier_end" in c]
            raw = fit.base.native_inputs(native,pcm)[0] if checks else None
            codes = []
            for c,old in zip(checks,a["checks"],strict=True):
                end = c["verifier_end"]
                assert end==old["end"]
                patch = np.rint(fit.room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
                q = outputs(vm,patch)
                assert fit.base.poll.digest(q)==old["models"]["student"]["output_codes_sha256"]
                codes.append(q)
            scores = matcher.scores(np.stack(codes)) if codes else []
            rows,accepted,joint = [],[],[]
            for c,s in zip(checks,scores,strict=True):
                at = c["verifier_end"]
                allow = s["score"] >= cal["threshold"]
                if allow:
                    accepted.append(at)
                if c["q8"] is not None and c["verifier_codes"][0]>=0:
                    joint.append(max(c["q8"],at))
                rows.append(dict(end=at,accepted=allow,**s))
            routes = {k:fit.base.scoring.merged(c161,ts) for k,ts in
                      (("paths",accepted),("paths_or_joint",accepted+joint))}
            records.append(dict(index=a["index"],pcm_sha256=a["pcm_sha256"],source=a["source"],key=a["key"],
                                spans=a["spans"],samples=len(pcm),c161=c161,checks=rows,routes=routes))
        save(RESULTS/f"{STEM}-{group}.json",dict(records=records,calibration_sha256=sha(CAL),
             letter_report_sha256=sha(letters_file),reference_sha256=sha(reference_file),script_sha256=sha(Path(__file__))))
        print(group,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("calibrate","evaluate"))
    {"calibrate":calibrate,"evaluate":evaluate}[p.parse_args().action]()
