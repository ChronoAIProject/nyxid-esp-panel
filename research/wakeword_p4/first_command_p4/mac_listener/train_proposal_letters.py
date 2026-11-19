"""Distill English CTC frame posteriors at training word-proposal clocks, then export a character-only INT8 net."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import proposal_ctc_evidence as reference
import train_positive_readout as positive
from export_c161_onset_evidence import transfer
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from score_proposal_readout import GROUPS

parent = positive.parent
room = parent.room
letters = room.BASE
base = parent.replay
STEM = "4090-c161-proposal-letters"
FIT = RESULTS / f"{STEM}-fit.json"
EXPORT = RESULTS / f"{STEM}-exports.json"
INITIAL = letters.weight_file("distilled")
TRAINED = RESULTS / f"{STEM}-student.npz"
SEED = 84751
STEPS = 650


def native_frontend():
    lib = base.BUILD/"frontend.so"
    assert sha(lib) == read(base.BUILD/"build.json")["library_sha256"]
    fn = ctypes.CDLL(str(lib.resolve())).ambient_history
    fn.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    return fn


def training_windows():
    inventory = {r["key"]:r for r in read(RESULTS/f"{parent.music.word.STEM}-training-v1.json")["training"]}
    native = native_frontend()
    front = RESULTS/f"{parent.music.STEM}-v1.tflite"
    assert sha(front) == read(RESULTS/f"{parent.music.STEM}-export-v1.json")["sha256"]
    proposer = REF_KERNEL(model_path=str(front))
    proposer.allocate_tensors()
    render = parent.music.mining.room_render()
    records, patches, clips = [], [], []
    for src, wave in parent.music.word.sources():
        info = inventory.pop(src["key"])
        assert base.poll.digest(wave) == info["pcm_sha256"] and src["text"] == info["text"]
        dry = np.pad(wave,(32000,16000))
        wet, setting = render(src,dry,32000)
        for name, pcm in (("dry",dry),("room",wet)):
            if name == "room" and np.array_equal(pcm,dry):
                continue
            raw, bands, clock = base.native_inputs(native,pcm)
            trace = base.infer(proposer,bands)
            events = base.scoring.word.times(base.scoring.word.decode(trace,clock))
            eligible = (clock[4:] >= 32000) & (clock[4:] <= 32000+len(wave))
            if src.get("end") is not None:
                eligible &= abs(clock[4:]-(32000+src["end"])) <= 3840
            choices = np.flatnonzero(eligible)
            assert len(choices)
            matches = [t for t in events if clock[4:][choices[0]] <= t <= clock[4:][choices[-1]]]
            peaks = np.convolve(trace.astype(np.int32),np.ones(5,np.int32),"valid")
            t = matches[0] if matches else int(clock[4:][choices[np.argmax(peaks[choices])]])
            end = t + base.scoring.word.WAIT
            patch = np.rint(room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
            clip = pcm[end-reference.WIDTH:end]
            assert len(clip) == reference.WIDTH
            records.append(dict(key=src["key"], voice=src["voice"], text=src["text"], label=letters.category(src["text"],src["positive"]),
                                kind=name, room=setting if name=="room" else None, source_pcm_sha256=info["pcm_sha256"],
                                stream_sha256=base.poll.digest(pcm), proposal=t, actual=bool(matches), end=end,
                                clip_sha256=base.poll.digest(clip), patch_sha256=base.poll.digest(patch)))
            patches.append(patch)
            clips.append(clip.copy())
    assert not inventory
    return np.stack(patches), clips, records, sha(front)


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    x, waves, records, front_hash = training_windows()
    teacher, proc, spec = letters.teacher()
    new_logits = []
    with torch.inference_mode():
        for i in range(0,len(waves),32):
            clips = [w.astype(np.float32)/32768 for w in waves[i:i+32]]
            samples = proc(clips,sampling_rate=16000,return_tensors="pt").input_values.cuda()
            new_logits.append(teacher(samples).logits.half().cpu().numpy())
    del teacher, waves
    cache = letters.WORK/"features.npz"
    assert sha(cache) == read(letters.SOURCES)["cache_sha256"]
    with np.load(cache) as data:
        anchors = len(data["labels"])
        labelled = np.r_[data["labels"],[r["label"] for r in records]]
        targets = torch.from_numpy(np.concatenate((data["logits"],np.concatenate(new_logits)))).cuda().float()
        inputs = torch.from_numpy(np.concatenate((data["x"],x))).cuda().unsqueeze(1)
    del new_logits
    blank = targets.argmax(2) == 0
    soft = (targets/2).softmax(2)
    nchar = targets.shape[-1]
    init = read(letters.FIT)["models"]["distilled"]["weights_sha256"]
    net = room.base.load(letters.Small(nchar),INITIAL,init).train()
    net.names.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00025,weight_decay=.002)
    rng = np.random.default_rng(SEED)
    positive_rows = anchors+np.flatnonzero(labelled[anchors:]==0)
    negative_rows = anchors+np.flatnonzero(labelled[anchors:]!=0)
    loss_log = []
    for step in range(STEPS):
        ids = np.r_[rng.integers(0,anchors,32),rng.choice(positive_rows,16),rng.choice(negative_rows,32)]
        _, frames = net(inputs[ids].float()/255)
        logits = frames[:,:-1]
        assert logits.shape == soft[ids].shape
        kl = F.kl_div((logits/2).log_softmax(2),soft[ids],reduction="none").sum(2)
        blanks = blank[ids]
        loss = 2*(kl[blanks].mean()+kl[~blanks].mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,99,STEPS-1):
            loss_log.append(dict(step=step+1,loss=loss.item()))
    with TRAINED.open("xb") as f:
        np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in net.state_dict().items()})
    save(FIT,dict(records=records, gpu=gpu, torch=torch.__version__, seed=SEED, steps=STEPS, teacher=spec,
             anchors=anchors, initial_sha256=init, weights_sha256=sha(TRAINED), front_sha256=front_hash,
             prior_cache_sha256=sha(cache), script_sha256=sha(Path(__file__)), history=loss_log,
             policy="Original training inventory, dry and existing training-room views. First actual proposal "
             "within 240 ms of a known word tail, else highest five-frame sum there. For unmarked negative "
             "sentences search the whole utterance. Distill 71 frame posteriors from the GPU CTC teacher "
             "in a 1.44 s PCM crop. Equal total loss weight on blank and nonblank frames. Retain original "
             "training-onset soft targets in each batch; no development PCM or transcripts used."))


def export():
    import tensorflow as tf
    tf.config.set_visible_devices([],"GPU")
    torch.set_num_threads(1)
    cfg = read(FIT)
    cache = letters.WORK/"features.npz"
    assert sha(cache) == cfg["prior_cache_sha256"]
    with np.load(cache) as z:
        rng = np.random.default_rng(SEED+1)
        ids = np.concatenate([rng.choice(np.flatnonzero(z["labels"]==k),32,replace=False) for k in range(5)])
        x = z["x"][ids].astype(np.float32)[:,:,:,None]/255
        alphabet = z["logits"].shape[-1]
    models = {}
    for key,path,digest in (("before",INITIAL,cfg["initial_sha256"]),("student",TRAINED,cfg["weights_sha256"])):
        net = room.base.load(letters.Small(alphabet),path,digest).cpu()
        clf = transfer(net)
        feature = next(layer.input for layer in clf.layers if isinstance(layer,tf.keras.layers.AveragePooling2D))
        conv = tf.keras.layers.Conv2D(alphabet,1)
        sequence = conv(feature)
        head = net.letters
        conv.set_weights([head.weight.detach().numpy().transpose(2,1,0)[:,None,:,:], head.bias.detach().numpy()])
        model = tf.keras.Model(clf.input, sequence)
        with torch.no_grad():
            expected = net(torch.from_numpy(x).permute(0,3,1,2))[1].numpy()
        got = np.concatenate([model(v[None],training=False).numpy()[:,:,0,:] for v in x])
        np.testing.assert_allclose(got,expected,atol=.00015,rtol=.0001)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: ([v[None]] for v in x)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = converter.inference_output_type = tf.int8
        blob = converter.convert()
        target = RESULTS/f"{STEM}-{key}.tflite"
        with target.open("xb") as f:
            f.write(blob)
        vm = REF_KERNEL(model_content=blob)
        vm.allocate_tensors()
        inp, = vm.get_input_details()
        out, = vm.get_output_details()
        assert inp["dtype"] == out["dtype"] == np.int8 and tuple(out["shape"]) == (1,72,1,alphabet)
        scale,zero = inp["quantization"]
        assert all(t["dtype"] in (np.int8,np.int32) for t in vm.get_tensor_details())
        agreements = 0
        for v,y in zip(x,expected,strict=True):
            vm.set_tensor(inp["index"],np.rint(v[None]/scale+zero).clip(-128,127).astype(np.int8))
            vm.invoke()
            ids_out = vm.get_tensor(out["index"])[0,:-1,0,:].argmax(1)
            agreements += int(np.count_nonzero(ids_out==y[:-1].argmax(1)))
        models[key] = dict(file=target.name,bytes=len(blob),sha256=sha(target),source_weights_sha256=digest,
                           argmax_equal=agreements,frames=len(x)*71,input_quantization=[scale,zero],
                           transfer_max_error=float(abs(got-expected).max()))
    save(EXPORT,dict(models=models,fit_sha256=sha(FIT),tensorflow=tf.__version__,representative_indices=ids.tolist()))


class CharacterModel:
    def __init__(self, model):
        self.vm = REF_KERNEL(model_path=str(model))
        self.vm.allocate_tensors()
        self.inp, = self.vm.get_input_details()
        self.out, = self.vm.get_output_details()
        scale,zero = self.inp["quantization"]
        self.table = np.rint(np.arange(256)/255/scale+zero).clip(-128,127).astype(np.int8)

    def infer(self, patch):
        self.vm.set_tensor(self.inp["index"],self.table[patch][None,:,:,None])
        self.vm.invoke()
        raw = self.vm.get_tensor(self.out["index"])[0,:-1,0,:]
        return raw.argmax(1), base.poll.digest(raw)


def evaluate():
    from transformers import Wav2Vec2CTCTokenizer
    from check_c161_pronunciation_ctc_v1 import MODEL as W2V
    proc = Wav2Vec2CTCTokenizer.from_pretrained(W2V,local_files_only=True)
    config = read(EXPORT)
    assert config["fit_sha256"] == sha(FIT)
    models = {}
    for k,s in config["models"].items():
        path = RESULTS/s["file"]
        assert sha(path) == s["sha256"]
        models[k] = CharacterModel(path)
    native = native_frontend()
    for group in GROUPS:
        reference_file = RESULTS/f"{reference.STEM}-{group}.json"
        records = []
        for (_,pcm,c161,_,_), r in zip(base.sources(group),read(reference_file)["records"],strict=True):
            assert base.poll.digest(pcm) == r["pcm_sha256"] and c161 == r["c161"]
            valid = [c for c in r["checks"] if "verifier_end" in c]
            raw = base.native_inputs(native,pcm)[0] if valid else None
            rows, accepted = [], {m:[] for m in models}
            joint = []
            for c in valid:
                at = c["verifier_end"]
                assert base.poll.digest(letters.cropped(pcm,at)) == c["neural_window"]["pcm_sha256"]
                patch = np.rint(room.window(raw,at).clip(0,666)*(255./666)).astype(np.uint8)
                trial = dict(proposal=c["proposal"],end=at,patch_sha256=base.poll.digest(patch),
                             teacher=c["neural_window"]["text"],models={})
                for m,vm in models.items():
                    ids,digest = vm.infer(patch)
                    text = proc.decode(ids.tolist())
                    decision = bool(reference.PHRASE.search(text))
                    trial["models"][m] = dict(text=text,accepted=decision,output_codes_sha256=digest)
                    if decision:
                        accepted[m].append(at)
                if c["q8"] is not None and c["verifier_codes"][0]>=0:
                    joint.append(max(c["q8"],at))
                rows.append(trial)
            routes = {m:base.scoring.merged(c161,events) for m,events in accepted.items()}
            routes.update({m+"_or_joint":base.scoring.merged(c161,events+joint) for m,events in accepted.items()})
            records.append(dict(index=r["index"],pcm_sha256=r["pcm_sha256"],samples=len(pcm),source=r["source"],
                                key=r.get("key"),spans=r.get("spans"),c161=c161,checks=rows,routes=routes))
        save(RESULTS/f"{STEM}-{group}.json",dict(records=records,exports_sha256=sha(EXPORT),
             reference_sha256=sha(reference_file), vocab_sha256=sha(W2V/"vocab.json"), script_sha256=sha(Path(__file__)),
             policy="Decode first 71 actual INT8 argmax letter frames with original greedy CTC and fixed phrase "
             "expression. Merge C161 and optionally Q8 with five-class confirmation. No score threshold sweep."))
        print(group,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("train","export","evaluate"))
    {"train":train,"export":export,"evaluate":evaluate}[p.parse_args().action]()
