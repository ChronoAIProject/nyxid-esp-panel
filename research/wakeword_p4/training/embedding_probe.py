"""Probe official pretrained embeddings on actual development WAVs, never classifier accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from openwakeword.utils import AudioFeatures

from stream_data import render_stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh report path")
    source = json.loads((a.models/"source.json").read_text())
    for row in source["feature_models"]:
        if hashlib.sha256((a.models/row["name"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Changed pretrained weights")
    model = AudioFeatures(melspec_model_path=str(a.models/"melspectrogram.onnx"),
                          embedding_model_path=str(a.models/"embedding_model.onnx"),
                          inference_framework="onnx", device=a.device, ncpu=2)
    providers = {"melspectrogram": model.melspec_model.get_providers(),
                 "embedding": model.embedding_model.get_providers()}
    if a.device == "gpu" and any(v[0] != "CUDAExecutionProvider" for v in providers.values()):
        raise RuntimeError("CUDA provider required; silent CPU fallback is not a GPU probe")
    # Upstream reset() fills the feature buffer from random noise. Replace that
    # history with embeddings from deterministic silence, before every stream.
    silence = model._get_embeddings(np.zeros(64000, dtype=np.int16))
    def streaming(pcm):
        model.reset()
        model.feature_buffer = silence.copy()
        values = []
        for at in range(0, len(pcm), 1280):
            block = pcm[at:at+1280]
            if len(block) < 1280:
                block = np.pad(block, (0, 1280-len(block)))
            model(block)
            values.append(model.get_features(16).copy())
        return np.concatenate(values)
    rows = list(map(json.loads, (a.features/"dataset/manifest.jsonl").read_text().splitlines()))
    selected = []
    for engine in sorted({r["engine"] for r in rows}):
        for label in (0, 1):
            selected.append(next(r for r in rows if r["split"] == "training" and r["engine"] == engine and r["label"] == label))
    results = []
    for row in selected:
        path = a.features/"dataset"/row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Changed source WAV")
        pcm, _ = render_stream(row, a.features/"dataset", np.random.default_rng(726), 0, [])
        pcm = np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)
        start = time.monotonic()
        first = streaming(pcm)
        seconds = time.monotonic()-start
        second = streaming(pcm)
        if not np.isfinite(first).all():
            raise ValueError("Nonfinite waveform-derived features")
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)
        results.append({"source": row["path"], "source_sha256": row["sha256"],
                        "engine": row["engine"], "label": row["label"],
                        "stream_window_shape": list(first.shape), "source_audio_seconds": len(pcm)/16000,
                        "stream_seconds": seconds, "reset_max_absolute_error": float(np.max(np.abs(first-second))),
                        "features_sha256": hashlib.sha256(first.tobytes()).hexdigest()})
    report = {"pretrained_source": source, "device": a.device,
              "provider": model.onnx_execution_provider, "clips": results,
              "model_providers": providers,
              "reset_reproducibility_passed": True,
              "startup": "Deterministic silence-derived embedding history replaces upstream random reset history.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Actual waveform-derived feature smoke probe only. No trained Hey Nyx classifier, recognition accuracy, INT8 MCU suitability or Hi ESP parity is established."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k,v in report.items() if k not in ("clips", "pretrained_source")}), flush=True)


if __name__ == "__main__":
    main()
