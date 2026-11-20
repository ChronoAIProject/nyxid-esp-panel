"""Fit a small causal acoustic check at C161's own five-score peaks and crossings."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import fit_c161_history_gain_v1 as base
import music_proposal_replay as replay
from train_name_final_s import integer_model, integer_scores
from train_proposal_letters import native_frontend

STEM = "4090-c161-event-patch"
WORK = Path(".local/wakeword/c161-event-patch")
FIT = base.RESULTS / (STEM + "-fit.json")
ALIGN = base.RESULTS / "4090-chatterbox-training-prefix-alignment-original-v1.json"
PHASES = (0, 240)
SEED = 20443
STEPS = 500


def corpus():
    bounds = {r["source"]["sha256"]:round(16000*r["automatic_prefix_end_seconds"])
              for r in base.read(ALIGN)["records"] if r["mechanically_usable"]}
    for row in base.sources():
        label = row["label"]
        # Piper's connected files use an isolated draw as a timing proxy. Fit its fully labeled bare calls.
        if row["source"]=="piper" and label and "," in row["text"]:
            continue
        pcm = base.audio(row)
        end = len(pcm)
        if row["source"]=="chatterbox" and label:
            end = bounds[row["sha256"]]
        yield {k:row[k] for k in ("source","voice","id","label","text","path","sha256")},pcm,end
    for row in base.read(base.RESULTS/base.DAILY)["admitted"]:
        req = row["request"]
        assert req["role"]=="training-only" and req["requested_label"]==0
        pcm = base.audio(row["wav"])
        yield dict(source="daily_training",voice=req["voice"],id=row["index"],label=0,text=req["text"],
                   path=row["wav"]["path"],sha256=row["wav"]["sha256"]),pcm,len(pcm)


def stream(native, pcm):
    raw,bands,clock = replay.native_inputs(native,pcm)
    values,_,_ = base.prepare(raw,base.PRIME+len(pcm))
    _,codes,_ = base.predict(values,str(base.MODEL),base.REF_KERNEL)
    total = np.convolve(codes.astype(np.int32),np.ones(5,np.int32),mode="full")[:len(codes)]
    return bands,clock,codes,total


def patches(bands, idx):
    # 24 native model calls * 30 ms. Average two adjacent calls and two adjacent frequency pairs.
    windows = bands[idx[:,None]+np.arange(-23,1)]
    return ((windows.astype(np.int32).reshape(len(idx),12,2,10,2).sum(axis=(2,4))+2)//4).astype(np.uint8).reshape(-1,120)


def collect():
    assert base.sha(base.MODEL)==base.MODEL_SHA
    native=native_frontend()
    records=[]
    features=[]
    seen={}
    for meta,pcm,end in corpus():
        previous=seen.get(meta["sha256"])
        if previous is not None:
            assert (previous["voice"],previous["label"])==(meta["voice"],meta["label"])
            previous["aliases"].append(meta)
            continue
        row=dict(meta,views=[],aliases=[])
        seen[meta["sha256"]]=row
        for phase in PHASES:
            start=24000+phase
            full=np.pad(pcm,(start,16000))
            bands,clock,codes,sums=stream(native,full)
            valid=np.flatnonzero((clock>start)&(clock<=start+end+(2400 if meta["label"] else 0)))
            maximum=valid[np.argmax(sums[valid])]
            # First threshold crossings sample the actual online decision. The strongest plausible window
            # is also kept for both labels, including negatives that never propose.
            ticks={int(maximum)}
            for fence in (960,1175):
                above=valid[sums[valid]>=fence]
                if len(above):
                    ticks.add(int(above[0]))
            ticks=np.asarray(sorted(ticks))
            xs=patches(bands,ticks)
            first=len(features)
            features.extend(xs)
            row["views"].append(dict(phase=phase,peak=int(sums[maximum]),clock=clock[ticks].tolist(),
                                     sums=sums[ticks].tolist(),ids=list(range(first,first+len(xs))),
                                     codes_sha256=sha256(codes.tobytes()).hexdigest()))
        records.append(row)
        if len(records)%60==0:
            print("training sources",len(records),flush=True)
    return records,np.stack(features)


def fit():
    device=torch.cuda.get_device_name()
    assert "4090" in device
    torch.set_num_threads(1)
    records,features=collect()
    # Calibrate by conditioning voice. Waveforms from one voice stay together across corpora and offsets.
    voices=sorted({r["voice"] for r in records},key=lambda v:sha256(v.encode()).digest())
    groups={v:("piper" if "/" in v else "chatterbox") for v in voices}
    calibration={next(v for v in voices if groups[v]=="chatterbox"), *[v for v in voices if groups[v]=="piper"][::5]}
    labels=np.array([r["label"] for r in records])
    fit_pools=[np.array([i for i,r in enumerate(records) if r["label"]==label and r["voice"] not in calibration])
               for label in (0,1)]
    x=torch.from_numpy(features).to("cuda",torch.float32)/255
    runs={}
    with TemporaryDirectory(dir=WORK) as tmp:
        folder=Path(tmp)
        for arm in ("nominal","shifted"):
            torch.manual_seed(SEED)
            rng=np.random.default_rng(SEED)
            net=nn.Sequential(nn.Linear(120,16),nn.ReLU(),nn.Linear(16,1)).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.002)
            losses=[]
            for step in range(STEPS):
                ids=np.r_[rng.choice(fit_pools[0],32),rng.choice(fit_pools[1],32)]
                phase_ids=rng.integers(len(PHASES),size=len(ids))
                draw=rng.random(len(ids))
                picks=[]
                for i,k,u in zip(ids,phase_ids,draw,strict=True):
                    view=records[i]["views"][k if arm=="shifted" else 0]["ids"]
                    picks.append(view[int(u*len(view))])
                logits=net(x[picks]).flatten()
                loss=F.binary_cross_entropy_with_logits(logits,torch.tensor(labels[ids],device="cuda",dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step in (0,99,499):
                    losses.append(float(loss.detach()))
            params,scale=integer_model(net)
            scores=integer_scores(features,params)
            with torch.inference_mode():
                error=float(np.max(np.abs(net(x).flatten().cpu().numpy()-scores*scale)))
            held=[i for r in records if r["voice"] in calibration and r["label"] for v in r["views"]
                  for i,s in zip(v["ids"],v["sums"],strict=True) if s>=960]
            assert held
            cut=int(scores[held].min())
            model=folder/(STEM+"-"+arm+".npz")
            np.savez_compressed(model,**params)
            runs[arm]=dict(weights=model.name,sha256=base.sha(model),scale=scale,cut=cut,calibration_windows=len(held),
                           calibration_positive_sources=sum(r["voice"] in calibration and r["label"]==1 and
                               any(v["peak"]>=960 for v in r["views"]) for r in records),losses=losses,logit_max_error=error)
            for r in records:
                for view in r["views"]:
                    view[arm]=scores[view["ids"]].tolist()
        feature_path=folder/(STEM+"-training-inputs.npz")
        np.savez_compressed(feature_path,features=features)
        base.save(folder/FIT.name,dict(seed=SEED,steps=STEPS,torch=torch.__version__,gpu=device,records=records,models=runs,
                  calibration_voices=sorted(calibration),phases=PHASES,features_sha256=base.sha(feature_path),
                  model_sha256=base.sha(base.MODEL),sources={p.name:base.sha(p) for p in (
                    Path(__file__),ALIGN,*(base.RESULTS/f for f in (base.PIPER,base.ADMISSION,base.BARE,base.DAILY)))},
                  policy="Source-balanced within each label. Native C161 peaks plus first five-score 960 and 1175 "
                         "crossings. Training-only Piper bare target/name, admitted Chatterbox wake and ordinary "
                         "speech. Same source draws and optimizer steps. Shifted arm selects one of two raw PCM "
                         "alignments per draw. Quantized lower cutoff preserves every qualifying calibration "
                         "positive window at both alignments in each arm."))
        for path in folder.iterdir():
            path.replace(base.RESULTS/path.name)
    print(runs,flush=True)


if __name__=="__main__":
    WORK.mkdir(parents=True,exist_ok=True)
    fit()
