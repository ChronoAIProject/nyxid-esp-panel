"""Compare literal spellings, competing names and command-initial sounds in Melo speech."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT/"results"
STEM = "4090-c161-melo-address-grammar"
PLAN = RESULTS/f"{STEM}-plan.json"
AUDIO = RESULTS/f"{STEM}-wav.zip"
WORK = Path(".local/wakeword/c161-melo-address-grammar")
RUNS = WORK/"willow"
NAMES = ("Hye Nyx", "Hey Nyx", "Hi Nicks", "Hey Nicks", "Hey Willow", "Hey Nick", "Hi Nick", "Hey Mick", "Hi Nikki", "Hey Max")
CARRIERS = ("switch the desk lamp off.", "open the garage window.")
DAILY = (("The switches are beside the door.", "Nick switched seats with me."),
         ("I think the garage is open.", "Mike and Chris are waiting outside."))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def generate():
    import torch
    from scipy.signal import resample_poly
    os.environ["NLTK_DATA"] = str(Path(".local/wakeword/melo-nltk-v1").resolve())
    sys.path.insert(0,str((ROOT/"generation").resolve()))
    from melo_english_adapter_v1 import EnglishFrontend, MODEL, load_acoustic_model

    if AUDIO.exists() or PLAN.exists():
        raise FileExistsError(STEM)
    WORK.mkdir(exist_ok=True)
    source = RESULTS/"4090-parallel-melo-context-generation-lock-v1.json"
    for p,h in read(source)["files"].items():
        assert sha(p) == h,p
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = read(MODEL/"config.json")
    assert config["data"]["sampling_rate"] == 44100 and config["data"]["hop_length"] == 512
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    frontend, model = EnglishFrontend(), load_acoustic_model(config)
    prior = read(RESULTS/"4090-c161-q8-piper-spellings-plan.json")
    files = {p:sha(p) for p in prior["runtime"].values()}
    assert all(h == prior["files"][p] for p,h in files.items())
    runner = ROOT/"qemu_hey_reference_v2"
    for p in (Path(__file__).resolve(),runner/"run_development.py", runner/"run_willow_mac_matched_prime_v1.py",
              ROOT/"experiments/run_c161_willow_matched_scenes_v1.py", ROOT/"experiments/check_c161_address_grammar.py"):
        files[str(p)] = sha(p)
    rows = []
    with ZipFile(AUDIO.with_suffix(".tmp"), "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for vi,voice in enumerate(("EN-US","EN-BR")):
            speaker = config["data"]["spk2id"][voice]
            for ci,(carrier,daily) in enumerate(zip(CARRIERS,DAILY,strict=True)):
                seed = 39641 + vi*2 + ci
                order = np.random.default_rng(seed).permutation(len(NAMES)+len(daily))
                # Same text order and per-address seed across the punctuation pair.
                for sep in (", ", " "):
                    inputs = [(name, name+sep+carrier) for name in NAMES] + [("daily",t) for t in daily]
                    chunks = [np.zeros(16000,np.int16)]
                    spans = []
                    at = len(chunks[0])
                    for label in order:
                        address,text = inputs[label]
                        request_seed = seed*100 + int(label)
                        random.seed(request_seed)
                        np.random.seed(request_seed)
                        torch.manual_seed(request_seed)
                        with torch.inference_mode():
                            info,(ph,ton,lang,bert,jbert) = frontend.tensors(text,config,"cuda:0")
                            y,attention,mask,_ = model.infer(ph,torch.tensor([ph.shape[-1]],device="cuda"),
                                torch.tensor([speaker],device="cuda"),ton,lang,bert,jbert,
                                sdp_ratio=.2,noise_scale=.6,noise_scale_w=.8,length_scale=1.,max_len=None)
                        native = y[0,0].cpu().numpy()
                        align = attention[0,0].cpu().numpy()
                        lengths = align.sum(axis=0).astype(np.int32)
                        assert len(native) == int(lengths.sum())*512 and np.isfinite(native).all()
                        assert np.all(align.sum(axis=1) == 1) and mask.shape[-1] == align.shape[0]
                        signal = resample_poly(native,160,441)
                        pcm = np.rint(signal*32768).clip(-32768,32767).astype("<i2")
                        spans.append(dict(address=address,text=text,start=at,end=at+len(pcm),seed=request_seed,
                            phones=info["phones"], normalized=info["normalized"], expanded_word2ph=info["expanded_word2ph"],
                            phone_ids=info["input_phone_ids"], token_duration_frames=lengths.tolist(),
                            native_pcm_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                            pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                            clipped=int(np.count_nonzero((signal < -1) | (signal > 32767/32768)))))
                        chunks.extend((pcm,np.zeros(32000,np.int16)))
                        at += len(pcm)+32000
                    pcm = np.concatenate(chunks)
                    path = WORK/f"{len(rows):02}.wav"
                    sf.write(path,pcm,16000,subtype="PCM_16")
                    archive.write(path,path.name)
                    files[str(path)] = sha(path)
                    rows.append(dict(voice=voice, carrier=carrier, separator=sep, target_id="paired",
                        path=str(path),sha256=sha(path),samples=len(pcm),
                        pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),timeline=spans))
                    print("completed",len(rows),voice,carrier,repr(sep),flush=True)
    AUDIO.with_suffix(".tmp").replace(AUDIO)
    spec = dict(records=rows, files=files, runtime=prior["runtime"], threshold_requests=["DEFAULT"],
                generator_lock_sha256=sha(source), archive_sha256=sha(AUDIO), torch_version=torch.__version__, gpu=gpu,
                rule="Two existing Melo development speaker conditions, new carriers with/without comma. Matched random seeds and order within each comma pair; daily utterances repeated. No speed change, output trimming, score filtering or threshold change.")
    PLAN.write_text(json.dumps(spec,indent=2)+"\n")


def score_melo():
    import check_c161_address_grammar as grammar
    grammar.evaluate(grammar.live_inputs(PLAN,"melo"),RESULTS/f"{STEM}.json",[PLAN])


def summarize_willow():
    import score_c161_piper_spellings as scoring
    from check_c161_pocketsphinx_first_command import load_reader
    reader, identity = load_reader()
    spec = read(PLAN)
    output = []
    uart = RESULTS/f"{STEM}-willow.zip"
    with ZipFile(uart.with_suffix(".tmp"),"w",ZIP_DEFLATED) as archive, patch.multiple(scoring,PLAN=PLAN,RUNS=RUNS):
        for i,src in enumerate(spec["records"]):
            pcm = scoring.load_wav(src)
            events = scoring.official(i,src,archive)
            data = scoring.annotate(src,pcm,dict(willow=events),[],[],reader)
            output.append(dict(index=i,pcm_sha256=src["pcm_sha256"],events=events,**data))
    uart.with_suffix(".tmp").replace(uart)
    scoring.save(RESULTS/f"{STEM}-willow.json",dict(records=output,asr=identity,plan_sha256=sha(PLAN),uart_archive_sha256=sha(uart)))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN = PLAN
        official.__file__ = str(Path(__file__).resolve())
        official.case()
    elif action == "willow":
        import run_c161_willow_matched_scenes_v1 as runner
        runner.PLAN = PLAN
        runner.OUT = RUNS
        runner.__file__ = str(Path(__file__).resolve())
        runner.willow()
    else:
        {"generate":generate,"score":score_melo,"summarize":summarize_willow}[action]()
