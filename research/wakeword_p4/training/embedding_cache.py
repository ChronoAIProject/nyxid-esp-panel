"""Cache official streaming embeddings from the existing development waveforms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from openwakeword.utils import AudioFeatures
import soundfile as sf

from features import features
from stream_data import render_stream


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class StreamingBackbone:
    def __init__(self, assets):
        self.source = json.loads((assets/"source.json").read_text())
        for row in self.source["feature_models"]:
            if sha(assets/row["name"]) != row["sha256"]:
                raise ValueError("Changed pretrained weights")
        self.model = AudioFeatures(melspec_model_path=str(assets/"melspectrogram.onnx"),
                                  embedding_model_path=str(assets/"embedding_model.onnx"),
                                  inference_framework="onnx", device="gpu", ncpu=2)
        for session in (self.model.melspec_model, self.model.embedding_model):
            session.set_providers([("CUDAExecutionProvider", {"use_tf32": "0"})])
        graph = onnx.load(assets/"melspectrogram.onnx")
        reductions = [n for n in graph.graph.node if n.op_type == "ReduceMax"]
        if len(reductions) != 1 or graph.opset_import[0].version != 13:
            raise ValueError("Unexpected pinned Mel graph")
        # The official graph clips log-Mel values against a global batch max.
        # Reduce all non-batch axes instead, exactly retaining N=1 semantics.
        reduction = reductions[0]
        del reduction.attribute[:]
        reduction.attribute.extend([onnx.helper.make_attribute("axes", [1, 2, 3]),
                                    onnx.helper.make_attribute("keepdims", 1)])
        nodes = []
        for node in graph.graph.node:
            if node.name == "Clip_28":
                # ONNX Clip requires scalar limits; Max/Min allow the equivalent
                # per-example limit to broadcast over time and frequency.
                temporary = node.output[0]+"_lower_bounded"
                nodes.extend([onnx.helper.make_node("Max", list(node.input[:2]), [temporary], name="PerExampleLowerBound"),
                              onnx.helper.make_node("Min", [temporary, node.input[2]], list(node.output), name="PerExampleUpperBound")])
            else:
                nodes.append(node)
        del graph.graph.node[:]
        graph.graph.node.extend(nodes)
        onnx.checker.check_model(graph)
        derived = assets/"melspectrogram-per-example-v2.onnx"
        data = graph.SerializeToString()
        if derived.exists() and derived.read_bytes() != data:
            raise ValueError("Existing derived graph differs; preserve it")
        if not derived.exists():
            derived.write_bytes(data)
        self.batch_mel = ort.InferenceSession(str(derived), providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])
        self.derived_mel_sha256 = sha(derived)
        self.providers = {"mel": self.model.melspec_model.get_providers(),
                          "embedding": self.model.embedding_model.get_providers(),
                          "batch_mel": self.batch_mel.get_providers()}
        if any(v[0] != "CUDAExecutionProvider" for v in self.providers.values()):
            raise RuntimeError("Both models must use CUDA")
        self.history = self.model._get_embeddings(np.zeros(64000, dtype=np.int16))[-15:].copy()

    def batch(self, waves):
        count = len(waves)
        steps = max((len(w)+1279)//1280 for w in waves)
        pcm = np.zeros((count, steps*1280), dtype=np.int16)
        for n, wave in enumerate(waves):
            pcm[n, :len(wave)] = wave
        mel = np.ones((count, 76, 32), dtype=np.float32)
        result = np.empty((count, steps, 96), dtype=np.float32)
        for step in range(steps):
            end = (step+1)*1280
            signal = pcm[:, max(0, end-1760):end].astype(np.float32)
            spec = self.batch_mel.run(None, {"input": signal})[0]/10+2
            spec = spec.reshape(count, -1, 32)
            mel = np.concatenate((mel, spec), axis=1)[:, -76:]
            result[:, step] = self.model.embedding_model_predict(mel[:, :, :, None]).reshape(count, 96)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite waveform-derived embeddings")
        return [np.concatenate((self.history, result[n, :(len(w)+1279)//1280])) for n,w in enumerate(waves)]

    def check(self, waves):
        batched = self.batch(waves)
        errors = []
        for wave, candidate in zip(waves, batched):
            self.model.reset()
            self.model.feature_buffer = self.history.copy()
            expected = []
            for at in range(0, len(wave), 1280):
                block = wave[at:at+1280]
                self.model(np.pad(block, (0, 1280-len(block))))
                expected.append(self.model.get_features(1)[0, -1])
            expected = np.asarray(expected)
            np.testing.assert_allclose(candidate[15:], expected, atol=1e-4, rtol=1e-4)
            errors.append(float(np.max(np.abs(candidate[15:]-expected))))
        return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--rirs", type=Path, required=True)
    p.add_argument("--rir-audit", type=Path, required=True)
    p.add_argument("--backgrounds", type=Path, required=True)
    p.add_argument("--librispeech", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, help="Smoke run only; cannot be used as a complete cache")
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    base = a.features.resolve()
    backbone = StreamingBackbone(a.models)
    rows = json.loads((base/"features.json").read_text())
    rirs = json.loads(a.rir_audit.read_text())["records"]
    for row in rirs:
        if sha(a.rirs/row["path"]) != row["sha256"]:
            raise ValueError("Changed RIR")
    bg = json.loads((a.backgrounds/"manifest.json").read_text())["records"]
    background_paths = {s: [str(a.backgrounds/r["path"]) for r in bg if r["split"] == s]
                        for s in ("training", "validation", "testing")}
    by_hash = {r["sha256"]: a.backgrounds/r["path"] for r in bg}
    for row in bg:
        if sha(a.backgrounds/row["path"]) != row["sha256"]:
            raise ValueError("Changed environment WAV")
    generated = [{**r, "kind": "generated", "estimated_samples": r["frames"]*160+320}
                 for r in rows if r["split"] in ("training", "validation")]
    negatives = []
    for name in ("human.json", "ambient.json"):
        for row in json.loads((base/name).read_text()):
            if row["split"] in ("training", "validation"):
                negatives.append({**row, "kind": "human" if "speaker" in row else "ambient",
                                  "estimated_samples": int(row["seconds"]*16000), "label": 0})
    jobs = sorted(generated+negatives, key=lambda r: r["estimated_samples"])
    if a.limit:
        # Stratified smoke cases exercise all source/label/augmentation branches.
        chosen = {}
        for row in jobs:
            key = (row["kind"], row.get("engine"), row.get("label"), row.get("variant"))
            chosen.setdefault(key, row)
        jobs = list(chosen.values())[:a.limit]
    raw = np.memmap(base/"features.bin", dtype="<u2", mode="r").reshape(-1, 40)
    def render(row):
        if row["kind"] == "generated":
            source = base/"dataset"/row["path"]
            if sha(source) != row["sha256"]:
                raise ValueError("Changed generated WAV")
            rng = np.random.default_rng(int(hashlib.sha256(row["path"].encode()).hexdigest()[:8], 16)+row["variant"])
            augmentation = None
            if row.get("augmentation") == "measured-rir-noise":
                rir = rirs[int(rng.integers(len(rirs)))]
                if rir["sha256"] != row["rir_sha256"]:
                    raise ValueError("RIR selection no longer reproduces")
                augmentation = {"rir_path": str(a.rirs/rir["path"]), "snr_db": row["snr_range_db"]}
            pcm, lead = render_stream(row, base/"dataset", rng, row["variant"], background_paths[row["split"]], augmentation)
            pcm = np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)
            end = (lead+row["keyword_end"])/16000 if row["label"] else None
            return row, pcm, end
        if row["kind"] == "human":
            parts = []
            for path in sorted((a.librispeech/str(row["speaker"])).rglob("*.flac")):
                wave, rate = sf.read(path, dtype="int16")
                if rate != 16000 or wave.ndim != 1:
                    raise ValueError("Invalid human WAV")
                parts.extend((wave, np.zeros(3200, dtype=np.int16)))
            pcm = np.concatenate(parts)
        else:
            source = base/row["audio_path"] if "audio_path" in row else by_hash[row["source_sha256"]]
            if sha(source) != row["source_sha256"]:
                raise ValueError("Changed ambient WAV")
            pcm, rate = sf.read(source, dtype="int16")
            if rate != 16000 or pcm.ndim != 1:
                raise ValueError("Invalid ambient WAV")
            pcm = np.resize(pcm, int(round(row["seconds"]*16000)))
        if len(pcm) != int(round(row["seconds"]*16000)):
            raise ValueError("Negative waveform reconstruction length mismatch")
        return row, pcm, None
    a.output.mkdir(parents=True)
    index, offset, checked, parity = [], 0, set(), []
    started = time.monotonic()
    with (a.output/"embeddings.bin").open("wb") as output, ThreadPoolExecutor(4) as pool:
        for at in range(0, len(jobs), a.batch):
            batch = list(pool.map(render, jobs[at:at+a.batch]))
            # Verify reconstructed generated audio against existing frontend bytes
            # once for every engine/label/variant, without using classifier scores.
            for row, pcm, _ in batch:
                key = (row.get("engine"), row["label"], row.get("variant"))
                if row["kind"] == "generated" and key not in checked:
                    actual = features(pcm)
                    expected = raw[row["offset"]:row["offset"]+row["frames"]]
                    np.testing.assert_array_equal(actual, expected)
                    checked.add(key)
            if not parity:
                parity = backbone.check([r[1] for r in batch])
            data = backbone.batch([r[1] for r in batch])
            for (row, pcm, end), value in zip(batch, data):
                output.write(value.astype("<f4").tobytes())
                index.append({**row, "offset": offset, "frames": len(value), "keyword_end_seconds": end,
                              "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()})
                offset += len(value)
            if at % (a.batch*10) == 0:
                print(json.dumps({"prepared": len(index), "planned": len(jobs), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    (a.output/"index.json").write_text(json.dumps(index)+"\n")
    report = {"complete_development_cache": not bool(a.limit), "sequences": len(index), "frames": offset,
              "feature_dim": 96, "history_frames": 15, "stride_seconds": .08,
              "pretrained_source": backbone.source, "providers": backbone.providers,
              "cuda_tf32_enabled": False,
              "derived_mel_sha256": backbone.derived_mel_sha256,
              "derived_mel_change": "ReduceMax axes=1,2,3 keepdims=1; dynamic Clip replaced by broadcast Max/Min. Per-example clipping preserves finite single-stream semantics.",
              "batch_vs_official_stream_max_errors": parity, "reconstructed_frontend_cases": len(checked),
              "embeddings_sha256": sha(a.output/"embeddings.bin"), "index_sha256": sha(a.output/"index.json"),
              "base_index_sha256": sha(base/"features.json"), "human_index_sha256": sha(base/"human.json"),
              "ambient_index_sha256": sha(base/"ambient.json"), "source_sha256": sha(Path(__file__)),
              "seconds": round(time.monotonic()-started, 1), "final_or_regression_audio_scored": False,
              "limits": "Frozen backbone desktop diagnostic only. Actual streaming waveform features; no classifier or MCU/Hi ESP equivalence yet."}
    (a.output/"cache.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("pretrained_source", "providers")}), flush=True)


if __name__ == "__main__":
    main()
