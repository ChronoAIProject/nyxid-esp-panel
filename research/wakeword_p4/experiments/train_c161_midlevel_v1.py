"""Train a quantized readout of an earlier causal history with word-end supervision."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from export_c161_midlevel_v1 import export
from fit_c161_history_gain_v1 import ADMISSION, BARE, DAILY, MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, read, save, sha
import generate_c161_confuser_readout_v1 as extra
import generate_c161_vowel_name_pair_v1 as pairs
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-midlevel"
PARENT = RESULTS / "4090-c161-paired-event-readout-speech-v1.tflite"
SEED = 47231
STEPS = 650


def sources():
    for plan, archive in ((pairs.SOURCE, pairs.AUDIO), (extra.SOURCE, extra.WAVS)):
        data = read(plan)
        assert sha(archive) == (data.get("pcm_sha256") or data["archive_sha256"])
        with np.load(archive) as waves:
            for r in data["records"]:
                if r.get("role", "fit") != "fit":
                    continue
                pcm = waves[r["key"]]
                assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
                positive = r["kind"] in ("nyx","lowered")
                end = r.get("prefix_end")
                if positive and end is None:
                    idx = r["phones"].index(",")
                    assert len(r["durations"]) == len(r["phones"])+2
                    end = sum(r["durations"][:1+idx])*400
                yield dict(key=plan.stem+"_"+r["key"],voice=r["voice"],positive=positive,text=r["text"],end=end),pcm
    for ledger in (ADMISSION, BARE, DAILY):
        for r in read(RESULTS/ledger)["admitted"]:
            request = r["request"]
            assert request["role"] == "training-only"
            positive = request["requested_label"] == 1
            end = round(r["automatic_prefix_end_seconds"]*16000) if positive else None
            yield dict(key=ledger+"_"+str(r["index"]),voice=request["voice"],positive=positive,text=request["text"],end=end),load_wav(r["wav"])


def trace(path, values, ids):
    it = REF_KERNEL(model_path=str(path),experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    inp, = it.get_input_details()
    s,z = inp["quantization"]
    q = np.clip(np.rint(values/s)+z,-128,127).astype(np.int8).reshape(-1,3,40)
    rows = [[] for _ in ids]
    for v in q:
        it.set_tensor(inp["index"],v[None])
        it.invoke()
        for buf, index in zip(rows,ids,strict=True):
            buf.append(it.get_tensor(index).reshape(-1))
    return [np.stack(v) for v in rows]


def collect():
    history = [r for r in read(RESULTS/DAILY)["admitted"] if r["request"]["group"]=="conversation"]
    features, teacher, masks, rows = [], [], [], []
    witnesses = []
    for i,(r,pcm) in enumerate(sources()):
        prev = history[(i+1)%len(history)]
        head = load_wav(prev["wav"])
        gap = (4000,8000,12000)[i%3]
        for condition, p in (("quiet",np.zeros_like(head)),("speech",head)):
            onset = len(head)+gap
            full = np.r_[p,np.zeros(gap,np.int16),pcm,np.zeros(8000,np.int16)]
            values, ck = frontend(full)
            h, logit = trace(PARENT,values,(58,63))
            if i in (0,1) and condition == "speech":
                witnesses.append(values)
            keep = ck>-1440
            h = h[keep]
            logit = logit[keep,0]
            ck = ck[keep]
            if r["positive"]:
                boundary = onset+r["end"]
                mask = (ck>=boundary-1440)&(ck<=boundary+3840)
            else:
                mask = np.ones(len(ck),bool)
            assert mask.sum() >= 5
            features.append(torch.from_numpy(h.copy()))
            teacher.append(torch.from_numpy(logit.copy()))
            masks.append(torch.from_numpy(mask))
            rows.append(dict(**r,condition=condition,onset=onset,history_id=prev["index"],samples=len(full),
                             pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),stream_sha256=hashlib.sha256(full.tobytes()).hexdigest()))
    def pad(x,value):
        return torch.nn.utils.rnn.pad_sequence(x,batch_first=True,padding_value=value).cuda()
    return pad(features,-128),pad(teacher,-128),pad(masks,False),rows,witnesses


def main():
    assert "4090" in torch.cuda.get_device_name()
    assert sha(MODEL)==MODEL_SHA and MODEL.stat().st_size==60952
    assert sha(PARENT)==read(RESULTS/"4090-c161-protected-confuser-training-v1.json")["parent_sha256"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    it=REF_KERNEL(model_path=str(PARENT),experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    quant={r["index"]:r["quantization"] for r in it.get_tensor_details()}
    sx,zx=quant[58]
    sl,zl=quant[63]
    raw,base,mask,rows,witnesses=collect()
    base=(base.float()-zl)*sl
    pos=np.flatnonzero([r["positive"] for r in rows])
    neg=np.flatnonzero([not r["positive"] for r in rows])
    w=torch.nn.Parameter(torch.randn(12,1260,device="cuda")*.012)
    b=torch.nn.Parameter(torch.full((12,),1.5,device="cuda"))
    v=torch.nn.Parameter(torch.linspace(-.25,.25,12,device="cuda"))
    bias=torch.nn.Parameter(torch.tensor(-3.,device="cuda"))
    opt=torch.optim.AdamW([w,b,v,bias],lr=.001,weight_decay=.0002)
    rng=np.random.default_rng(SEED)
    def rounded(x,scale):
        q=(x/scale).round()*scale
        return x+(q-x).detach()
    def forward(x,old):
        ww=rounded(w,w.detach().abs().amax()/127)
        h=rounded((F.linear(x,ww,b)).clamp(0,6),6/255)
        vote=rounded((old+3).clamp(0,6),6/255)
        weights=torch.cat((v,v.new_ones(1)))
        weights=rounded(weights,weights.detach().abs().amax()/127)
        predicted=h@weights[:-1]+vote*weights[-1]+bias
        return rounded(predicted.clamp((-128-zl)*sl,(127-zl)*sl),sl)
    history=[]
    for step in range(STEPS):
        ids=np.r_[rng.choice(pos,24),rng.choice(neg,32)]
        y=forward((raw[ids].float()-zx)*sx,base[ids])
        valid=mask[ids].unfold(1,5,1).all(-1)
        window=y.unfold(1,5,1)
        quality=window.sigmoid().mean(-1).masked_fill(~valid,-1)
        loc=quality.detach().argmax(1)
        chosen=window[torch.arange(len(ids),device="cuda"),loc]
        absolute=F.softplus(4.2-chosen[:24]).mean()+F.softplus(chosen[24:]+1.8).mean()
        # Keep the strongest physically aligned parent response on positive recordings.
        teacher=base[ids[:24]].unfold(1,5,1)
        tloc=teacher.sigmoid().mean(-1).masked_fill(~valid[:24],-1).argmax(1)
        idx=torch.arange(24,device="cuda")
        floor=(teacher[idx,tloc].clamp(max=4.2)-window[:24][idx,tloc]).relu().mean()
        random_loc=(torch.rand_like(quality[24:]).masked_fill(~valid[24:],-1)).argmax(1)
        random_neg=window[24:][torch.arange(32,device="cuda"),random_loc]
        loss=absolute+.3*floor+.25*F.softplus(random_neg+2).mean()
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,99,299,649):
            history.append(dict(step=step+1,loss=loss.item(),positive=chosen[:24].min(-1).values.mean().item(),
                                negative=chosen[24:].mean().item()))
    arr=lambda x:x.detach().cpu().numpy()
    dest=RESULTS/f"{STEM}-single-v1.tflite"
    hidden,vote,last=export(PARENT,dest,arr(w),arr(b),np.r_[arr(v),1.].astype(np.float32)[None],np.array([bias.item()],np.float32))
    errors=[]
    for values in witnesses:
        native,old,actual=trace(dest,values,(58,63,last))
        prediction=forward((torch.tensor(native,device="cuda").float()-zx)*sx,
                           (torch.tensor(old[:,0],device="cuda").float()-zl)*sl)
        delta=abs(np.rint(arr(prediction)/sl)+zl-actual[:,0])
        errors.append(dict(frames=len(delta),different=int((delta>0).sum()),max_logit_codes=int(delta.max())))
        assert delta.max()<=2,errors
    rt=REF_KERNEL(model_path=str(dest))
    rt.allocate_tensors()
    inp,=rt.get_input_details()
    out,=rt.get_output_details()
    assert list(inp["shape"])==[1,3,40] and out["dtype"]==np.uint8 and out["quantization"]==(1/256,0)
    ledger=RESULTS/f"{STEM}-sources-v1.json"
    save(ledger,dict(records=rows))
    save(RESULTS/f"{STEM}-training-v1.json",dict(gpu=torch.cuda.get_device_name(),baseline_sha256=MODEL_SHA,parent_sha256=sha(PARENT),
         positive=len(pos),negative=len(neg),steps=STEPS,seed=SEED,history=history,sources_sha256=sha(ledger),export_checks=errors,
         reports=[dict(name="verifier",path=dest.name,bytes=dest.stat().st_size,sha256=sha(dest))]))


if __name__=="__main__":
    main()
