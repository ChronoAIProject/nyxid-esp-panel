"""GPU-synthesize matched commands and everyday lexical conflicts on training-only voices."""
import hashlib

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

from pathlib import Path
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-confuser-readout"
WAVS = RESULTS / f"{STEM}-pcm-v1.npz"
SOURCE = RESULTS / f"{STEM}-sources-v1.json"
VOICES = ("af_alloy","am_michael","bf_isabella","am_onyx","af_jessica","bm_fable")
CARRIERS = ("please close the pantry.", "move the small chair.", "set my morning alarm.")
DAILY = ("We need six mugs for breakfast.", "Please bring six blankets for our guests.",
         "Please thank Kevin for the tickets.", "Our best students finished early.",
         "These six seats are still empty.", "Can you send the next invoice?")


def main():
    plan = read(RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    approved = {r["request"]["conditioning_id"] for r in plan["requests"] if r["request"]["split"]=="training"}
    assert set(VOICES) <= approved
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p)==plan["files"][str(p)] for p in (model,bank))
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    opts.enable_profiling = True
    opts.profile_file_prefix = ".local/wakeword/c161-confuser-synthesis-profile"
    session = ort.InferenceSession(str(model),opts,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    engine = Kokoro.from_session(session,str(bank))
    sentences = [(kind, prefix + ", " + command, prefix)
                 for command in CARRIERS for kind,prefix in (("nyx","Hey Nicks"),("nick","Hey Nick"),("nicole","Hey Nicole"))]
    sentences += [("daily",text,"") for text in DAILY]
    rows, audio, counts = [], {}, {}
    for voice in VOICES:
        for speed in (.9,1.1):
            for kind,text,prefix in sentences:
                lang = "en-gb" if voice.startswith("b") else "en-us"
                phones = engine.tokenizer.phonemize(text,lang)
                ids = engine.tokenizer.tokenize(phones)
                packed = np.array([[0,*ids,0]],np.int64)
                style = np.asarray(engine.get_voice_style(voice)[len(ids)],np.float32)
                raw,dur = session.run(None,dict(input_ids=packed,style=style,speed=np.array([speed],np.float32)))
                if not counts:
                    import json
                    for op in json.loads(Path(session.end_profiling()).read_bytes()):
                        provider = op.get("args",{}).get("provider")
                        if provider:
                            counts[provider] = counts.get(provider,0) + 1
                    assert counts.get("CUDAExecutionProvider",0)>0
                signal = resample_poly(raw.ravel(),2,3)
                assert np.isfinite(signal).all() and len(signal) == int(dur.sum())*400
                pcm = np.rint(np.clip(signal,-1,32767/32768)*32768).astype("<i2")
                k = str(len(rows))
                audio[k] = pcm
                rows.append(dict(key=k,kind=kind,text=text,voice=voice,speed=speed,phones=phones,
                                 clipped=int(((signal< -1)|(signal>32767/32768)).sum()),durations=dur.ravel().tolist(),
                                 pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    assert len(rows) == 180
    with WAVS.open("xb") as f:
        np.savez_compressed(f,**audio)
    save(SOURCE,dict(model_sha256=sha(model),bank_sha256=sha(bank),archive_sha256=sha(WAVS),kernels=counts,records=rows))


if __name__=="__main__":
    main()
