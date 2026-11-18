"""Correct the one-output contextual C161 candidate with full-sequence negative mining."""
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha
from generate_c161_vowel_name_pair_v1 import AUDIO, SOURCE
from replay_c161_ambient_history_v1 import BUILD
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_paired_event_readout_v1 import both_histories, DAILY

STEM = "4090-c161-bounded-readout"
PARENT = RESULTS / "4090-c161-paired-event-readout-speech-v1.tflite"
SEED = 5421


def inputs():
    ledger = read(RESULTS / "4090-multispeaker-coverage-native-original-v1.json")
    path = Path(ledger["records_path"])
    assert sha(path) == ledger["records_sha256"]
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["length_scale"] == 1 and r["draw"] == 0 and r["text"] in ("Hey Nicks.", "Hey Nick."):
            assert r["role"] == "training"
            pcm = load_wav(dict(path=path.parent/r["path"], sha256=r["sha256"]))
            yield r["id"], r["voice"] + "/" + r["speaker_name"], r["text"] == "Hey Nicks.", pcm
    src = read(SOURCE)
    assert sha(AUDIO) == src["pcm_sha256"]
    with np.load(AUDIO) as waves:
        for r in src["records"]:
            if r["role"] == "fit" and r["kind"] in ("nyx", "lowered", "nick", "nicole"):
                pcm = waves[r["key"]]
                assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
                yield "kokoro-" + r["key"], r["voice"], r["kind"] in ("nyx", "lowered"), pcm
    for r in read(DAILY)["admitted"]:
        if r["request"]["group"] in ("name_address", "number_quantity"):
            assert r["request"]["role"] == "training-only"
            yield "daily-" + str(r["index"]), r["request"]["voice"], False, load_wav(r["wav"])


def collect():
    path = BUILD / "frontend.so"
    assert sha(path) == read(BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(path.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    histories = [r for r in read(DAILY)["admitted"] if r["request"]["group"] == "conversation"]
    records, cache = [], []
    for i,(key,voice,label,target) in enumerate(inputs()):
        history = histories[i % len(histories)]
        prefix = load_wav(history["wav"])
        gap = (4000,8000,16000)[i % 3]
        onset = len(prefix) + gap
        for condition, head in (("silence", np.zeros_like(prefix)), ("speech", prefix)):
            pcm = np.r_[head, np.zeros(gap,np.int16), target, np.zeros(8000,np.int16)]
            for arm,(clock,hidden,_) in enumerate(both_histories(pcm,native)):
                # Include target region and trailing silence, with history already integrated.
                cache.append(torch.as_tensor(hidden[clock>onset].copy(),device="cuda"))
                records.append(dict(key=key,voice=voice,positive=label,condition=condition,arm=arm,gap=gap,
                                    onset=onset, history=history["index"], target_pcm_sha256=hashlib.sha256(target.tobytes()).hexdigest()))
    return records, cache


def train():
    assert "4090" in torch.cuda.get_device_name() and sha(MODEL) == MODEL_SHA
    expected = read(RESULTS/"4090-c161-paired-event-readout-training-v1.json")
    assert sha(PARENT) == next(r["sha256"] for r in expected["reports"] if r["name"]=="speech")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    it = REF_KERNEL(model_path=str(PARENT), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    quant = {r["index"]:r["quantization"] for r in it.get_tensor_details()}
    sx,zx = quant[62]
    sw,zw = quant[14]
    assert zx == -128 and zw == 0
    start = expected["weight_offset"]
    orig = it.get_tensor(14).ravel()
    blob = PARENT.read_bytes()
    assert blob[start:start+len(orig)] == orig.tobytes()
    teacher = torch.as_tensor(orig,device="cuda",dtype=torch.float32) * sw
    bias = int(it.get_tensor(13)[0]) * sx * sw
    records, features = collect()
    lengths = torch.tensor([len(f) for f in features],device="cuda")
    x = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=-128)
    del features
    positive = np.array([r["positive"] for r in records])
    pi,ni = np.flatnonzero(positive),np.flatnonzero(~positive)
    def peaks(indices,w):
        logits = ((x[indices].float() - zx) * sx) @ w + bias
        # Train on strongest contiguous five-frame probability, matching streaming smoothing.
        mean = logits.sigmoid().unfold(1,5,1).mean(-1)
        valid = torch.arange(mean.size(1),device="cuda")[None] + 5 <= lengths[indices,None]
        return mean.masked_fill(~valid,-1).amax(dim=1)
    parent_peaks = torch.cat([peaks(ids,teacher) for ids in np.array_split(np.arange(len(x)),64)])
    ledger = RESULTS/f"{STEM}-sources-v1.json"
    save(ledger,dict(records=records,parent_peaks=parent_peaks.cpu().tolist()))
    reports=[]
    for name in ("free","bounded"):
        rng=np.random.default_rng(SEED)
        w=teacher.clone().requires_grad_()
        opt=torch.optim.Adam([w],lr=.00007)
        history=[]
        for step in range(240):
            pos=rng.choice(pi,32)
            neg=rng.choice(ni,48)
            rounded=(w/sw).round().clamp(-127,127)*sw
            wq=w+(rounded-w).detach()
            pos_peak,neg_peak=peaks(pos,wq),peaks(neg,wq)
            retain=(parent_peaks[pos]-pos_peak).relu().square().mean()
            # Minimize the worst eight negatives in this batch with a margin below threshold.
            conflict=(neg_peak.topk(8).values-.70).relu().square().mean()
            loss=4*retain+conflict+.005*((wq-teacher)/sw).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if name=="bounded":
                    w.copy_(torch.minimum(w,teacher))
            if step in (0,59,119,239):
                history.append(dict(step=step+1,retention=retain.item(),conflict=conflict.item(),loss=loss.item()))
        final=(w.detach()/sw).round().clamp(-127,127).to(torch.int8).cpu().numpy()
        if name=="bounded":
            assert np.all(final<=orig)
        out=RESULTS/f"{STEM}-{name}-v1.tflite"
        with out.open("xb") as f:
            f.write(blob[:start]+final.tobytes()+blob[start+len(orig):])
        check=REF_KERNEL(model_path=str(out))
        check.allocate_tensors()
        np.testing.assert_array_equal(check.get_tensor(14).ravel(),final)
        assert len(check.get_input_details())==len(check.get_output_details())==1
        reports.append(dict(name=name,path=out.name,sha256=sha(out),bytes=out.stat().st_size,
                            up=int((final>orig).sum()),down=int((final<orig).sum()),history=history))
    save(RESULTS/f"{STEM}-training-v1.json",dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
         baseline_sha256=MODEL_SHA,parent_sha256=sha(PARENT),ledger_sha256=sha(ledger),seed=SEED,steps=240,reports=reports))


if __name__=="__main__":
    if sys.argv[1:]==["train"]:
        train()
    else:
        import replay_c161_vowel_name_pair_v1 as replay
        replay.STEM=STEM
        for group in sys.argv[1:]:
            replay.run(group)
