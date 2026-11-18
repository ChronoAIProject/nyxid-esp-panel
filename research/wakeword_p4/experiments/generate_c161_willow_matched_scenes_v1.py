"""Prospective six matched synthetic scenes per previously consumed US/BR condition."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT / "results"
OUT = Path(".local/wakeword/c161-willow-matched-scenes-v1")
STEM = "4090-c161-willow-matched-scenes"
SCENES = [
    ("The bus arrived before the bakery opened.", "Hey Nick, can you move your bicycle?", "read out my next appointment."),
    ("I put the blue towels on the shelf.", "Nicole, the back gate is open.", "set a timer for the cookies."),
    ("They moved the picnic indoors because of the wind.", "Hey Will, bring the plates when you come.", "switch on the small desk light."),
    ("The post office closes early today.", "Willow, your lunch is on the table.", "tell me when the train leaves."),
    ("Someone left a scarf on the chair.", "Hey Nicole, have you seen the new menu?", "turn down the radio in the kitchen."),
    ("We have enough flour for one more loaf.", "Nick, please pass me the clean bowl.", "add black tea to my shopping list."),
]
os.environ["NLTK_DATA"] = str(Path(".local/wakeword/melo-nltk-v1").resolve())
sys.path.insert(0,str((ROOT / "generation").resolve()))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def generate():
    import torch
    from melo_english_adapter_v1 import EnglishFrontend, MODEL, load_acoustic_model
    previous = RESULTS / "4090-parallel-melo-context-generation-lock-v1.json"
    frozen = json.loads(previous.read_bytes())
    for name,h in frozen["files"].items():
        assert digest(Path(name))==h,name
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    config = json.loads((MODEL/"config.json").read_bytes())
    assert config["data"]["sampling_rate"] == 44100
    frontend, model = EnglishFrontend(), load_acoustic_model(config)
    OUT.mkdir(exist_ok=False)
    rows, sessions = [], []
    for voice in ["EN-US","EN-BR"]:
        speaker = config["data"]["spk2id"][voice]
        for i,(chat,name,command) in enumerate(SCENES):
            waves = {}
            for role,text in [("chat",chat),("name",name),("nyx","Hey Nicks, "+command),("willow","Hey Willow, "+command)]:
                seed = 741 + i
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    info,(phones,tones,languages,bert,ja_bert) = frontend.tensors(text,config,"cuda:0")
                    output,attention,mask,_ = model.infer(phones,torch.tensor([phones.shape[-1]],device="cuda:0"),
                        torch.tensor([speaker],device="cuda:0"),tones,languages,bert,ja_bert,sdp_ratio=.2,
                        noise_scale=.6,noise_scale_w=.8,length_scale=1.0,max_len=None)
                native = output[0,0].cpu().numpy()
                assert np.isfinite(native).all() and len(native)==attention.shape[2]*512 and mask.shape[-1]==attention.shape[2]
                target = OUT/f"{voice}-{i}-{role}"
                np.save(target.with_suffix(".npy"),native,allow_pickle=False)
                aligned = target.with_suffix(".npz")
                np.savez_compressed(aligned,attention=attention.cpu().numpy(),mask=mask.cpu().numpy())
                converted = resample_poly(native,160,441)
                pcm = np.rint(np.clip(converted,-1,32767/32768)*32768).astype(np.int16)
                sf.write(target.with_suffix(".wav"),pcm,16000,subtype="PCM_16")
                waves[role] = pcm
                rows.append(dict(voice=voice,scene=i,role=role,text=text,seed=seed,phones=info,samples=len(pcm),
                                 native_sha256=digest(target.with_suffix(".npy")),alignment_sha256=digest(aligned),
                                 wav_sha256=digest(target.with_suffix(".wav")),clipped=int(np.count_nonzero((converted<-1)|(converted>32767/32768)))))
            prefix = np.concatenate((waves["chat"],np.zeros(8000,np.int16),waves["name"],np.zeros(8000,np.int16)))
            for role in ["nyx","willow"]:
                pcm = np.concatenate((prefix,waves[role]))
                path = OUT/f"{voice}-{i}-{role}-stream.wav"
                sf.write(path,pcm,16000,subtype="PCM_16")
                sessions.append(dict(voice=voice,scene=i,target_id=role,carrier=command,path=str(path),samples=len(pcm),
                    sha256=digest(path),pcm_sha256=hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest(),
                    prefix_pcm_sha256=hashlib.sha256(prefix.astype("<i2").tobytes()).hexdigest(),
                    chat_end=len(waves["chat"]),name_start=len(waves["chat"])+8000,name_end=len(prefix)-8000,target_start=len(prefix)))
    assert len(rows)==48 and len(sessions)==24
    result = dict(gpu=torch.cuda.get_device_name(),source_lock_sha256=digest(previous),rows=rows,records=sessions)
    with (RESULTS/f"{STEM}-sources-v1.json").open("x") as f:
        json.dump(result,f,indent=2)
        f.write("\n")


if __name__=="__main__":
    generate()
