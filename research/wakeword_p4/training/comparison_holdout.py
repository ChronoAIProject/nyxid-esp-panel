"""Freeze paired wake phrases with unused Kokoro international voice presets.

No model is loaded or scored. This is synthetic English rendered through
international carrier voices, not a representative human-speaker benchmark.
Common negative phrases exclude both target wake words.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kokoro_onnx import Kokoro
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf

from kokoro_samples import MODEL_SHA, VOICES_SHA


class Session(ort.InferenceSession):
    requested_speed = 1.0

    def run(self, names, feed, run_options=None):
        schema = next(v for v in self.get_inputs() if v.name == "speed")
        if schema.type != "tensor(float)":
            raise ValueError("This pinned comparison protocol requires float speed")
        feed = {**feed, "speed": np.asarray([self.requested_speed], dtype=np.float32)}
        return super().run(names, feed, run_options)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Use a fresh output directory")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, digest in (("kokoro-v1.0.onnx", MODEL_SHA), ("voices-v1.0.bin", VOICES_SHA)):
        if hashlib.sha256((a.assets/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Asset checksum mismatch: {name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = Session(str(a.assets/"kokoro-v1.0.onnx"), options, providers=["CPUExecutionProvider"])
    engine = Kokoro.from_session(session, str(a.assets/"voices-v1.0.bin"))
    voices = sorted(v for v in engine.get_voices() if not v.startswith(("af_", "am_", "bf_", "bm_")))
    plan = [("Hey Nicks.", "hey_nyx"), ("Hi E S P.", "hi_esp"),
            ("Hey Nick.", None), ("Hey Next.", None), ("Hey Mix.", None),
            ("Please check the volume.", None)]
    a.output.mkdir(parents=True)
    records = []
    with (a.output/"manifest.jsonl").open("w") as log:
        for voice in voices:
            for speed in (.9, 1., 1.1):
                session.requested_speed = speed
                for index, (text, target) in enumerate(plan):
                    pcm, rate = engine.create(text, voice=voice, speed=speed, lang="en-us")
                    if rate != 24000:
                        raise ValueError("Expected 24 kHz synthesis")
                    pcm = resample_poly(pcm, 2, 3)
                    path = Path(voice)/f"{speed:.1f}-{index}.wav"
                    (a.output/path).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(a.output/path, pcm, 16000, subtype="PCM_16")
                    row = {"path": str(path), "text": text, "target": target,
                           "phonemes": engine.tokenizer.phonemize(text, "en-us"),
                           "voice": voice, "speed": speed, "samples": len(pcm),
                           "sha256": hashlib.sha256((a.output/path).read_bytes()).hexdigest()}
                    records.append(row)
                    log.write(json.dumps(row)+"\n"); log.flush()
            print(json.dumps({"voice": voice, "clips": len(records)}), flush=True)
    report = {"protocol_version": 2, "voice_ids": voices, "clips": len(records), "model_sha256": MODEL_SHA,
              "voices_sha256": VOICES_SHA, "generator_sha256": source_hash,
              "manifest_sha256": hashlib.sha256((a.output/"manifest.jsonl").read_bytes()).hexdigest(),
              "targets": {str(t): sum(r["target"] == t for r in records) for t in ("hey_nyx", "hi_esp", None)},
              "scores_computed": False,
              "protocol": "Paired Hey Nicks/Hi ESP with same voice/speed; common negatives omit both keywords. Voice IDs are disjoint from all English presets used in development; freeze before scoring any candidate.",
              "limits": "Same Kokoro model, unused international voice IDs rendering English. This is a synthetic accent challenge, not independent human speech or cross-generator validation. Preset IDs do not establish physical-speaker identity. Model inference/ASR/human pronunciation review and matched acoustic rendering are still required before final comparison."}
    (a.output/"generation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
