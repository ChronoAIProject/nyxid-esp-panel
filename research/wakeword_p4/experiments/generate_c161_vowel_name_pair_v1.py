"""GPU Kokoro minimal pairs for a fixed C161 readout intervention."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT / "results"
OUT = Path(".local/wakeword/c161-vowel-name-v1")
AUDIO = RESULTS / "4090-c161-vowel-name-pcm-v1.npz"
SOURCE = RESULTS / "4090-c161-vowel-name-sources-v1.json"
VOICES = ("af_bella", "am_onyx", "bf_isabella", "bm_fable", "af_jessica", "am_echo", "af_nova", "bm_lewis")
PREFIX = {"nyx": "Hey Nicks", "lowered": "Hey Nicks", "nick": "Hey Nick", "nicole": "Hey Nicole", "next": "Hey Next"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    import subprocess
    assert "4090" in subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    plan = json.loads((RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json").read_bytes())
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert sha(model) == plan["files"][str(model)] and sha(bank) == plan["files"][str(bank)]
    approved = {r["request"]["conditioning_id"] for r in plan["requests"] if r["request"]["split"] == "training"}
    assert set(VOICES) <= approved
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    OUT.mkdir(exist_ok=True, parents=True)
    opts.profile_file_prefix = str(OUT / "synthesis")
    # Profile one real request. Avoid retaining a large ORT operator log for every draw.
    session = ort.InferenceSession(str(model), sess_options=opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    wrapper = Kokoro.from_session(session, str(bank))
    pcm, rows = {}, []
    kernels = None
    for voice in VOICES:
        lang = "en-gb" if voice.startswith("b") else "en-us"
        for speed in (0.9, 1.1):
            items = [(kind, base + suffix, base) for suffix in (".", ", open the kitchen door.", ", can you bring the blue cup?")
                     for kind, base in PREFIX.items()]
            items.extend([("daily_next", "What's next on the list?", ""), ("daily_nicole", "Nicole has the green folder.", "")])
            for kind, text, base in items:
                phones = wrapper.tokenizer.phonemize(text, lang)
                prefix = wrapper.tokenizer.phonemize(base, lang) if base else ""
                if kind == "lowered":
                    assert prefix.endswith("ɪks") and phones.startswith(prefix)
                    phones = prefix[:-3] + "ɛks" + phones[len(prefix):]
                    prefix = prefix[:-3] + "ɛks"
                tokens = wrapper.tokenizer.tokenize(phones)
                prefix_ids = wrapper.tokenizer.tokenize(prefix) if prefix else []
                assert tokens[:len(prefix_ids)] == prefix_ids and len(tokens) == len(phones)
                padded = np.array([[0, *tokens, 0]], dtype=np.int64)
                style = np.asarray(wrapper.get_voice_style(voice)[len(tokens)], dtype=np.float32)
                raw, dur = session.run(None, dict(input_ids=padded, style=style, speed=np.array([speed], np.float32)))
                if kernels is None:
                    profile = Path(session.end_profiling())
                    kernels = {}
                    for event in json.loads(profile.read_bytes()):
                        ep = event.get("args", {}).get("provider")
                        if ep:
                            kernels[ep] = kernels.get(ep, 0) + 1
                    assert kernels.get("CUDAExecutionProvider", 0) > 0
                dur = dur.ravel()
                assert len(dur) == padded.size and raw.size == int(dur.sum()) * 600
                raw = raw.ravel()
                audio = resample_poly(raw, 2, 3)
                assert np.isfinite(audio).all()
                code = np.rint(np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2")
                assert np.any(code)
                key = str(len(rows))
                pcm[key] = code
                rows.append(dict(key=key, voice=voice, role="fit" if voice in VOICES[:6] else "transfer",
                                 speed=speed, kind=kind, text=text, phones=phones, prefix=prefix, tokens=padded.ravel().tolist(),
                                 duration=dur.tolist(), prefix_end=int(dur[:1+len(prefix_ids)].sum()) * 400 if base else None,
                                 samples=len(code), clipped=int(np.count_nonzero((audio < -1) | (audio > 32767 / 32768))),
                                 native_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                                 pcm_sha256=hashlib.sha256(code.tobytes()).hexdigest()))
        print(voice, len(rows), flush=True)
    assert len(rows) == 272
    with AUDIO.open("xb") as f:
        np.savez_compressed(f, **pcm)
    with SOURCE.open("x") as f:
        json.dump(dict(model_sha256=sha(model), bank_sha256=sha(bank), pcm_sha256=sha(AUDIO),
                       first_request_kernel_counts=kernels, ort=ort.__version__, records=rows), f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
