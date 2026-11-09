"""Prepare checksum-pinned Mandarin FLEURS negatives without scoring a model.

Keep the official validation/test partitions. Concatenate exact normal-speed
PCM with explicit separators; use one frontend per resulting WAV. Individual
speaker IDs are not available in the exported schema and are never invented.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from continuous_negatives import ALIASES
from features import continuous_wav_features

REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
SHARDS = {
    "validation": (287985961, "18698ffdd46c36c54af641821684d3f0313a7b64e0a49615597ec61a98f2b57e"),
    "test": (695674033, "87c0aebbe183f3a36ac87b5c3421b6ab57036824744ff695029a3f858e7622fd"),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def feature_worker(job):
    root, record = job
    root = Path(root)
    raw, padding = continuous_wav_features(root/record["pcm_path"])
    feature_path = root/"features"/f"{record['id']}.npy"
    np.save(feature_path, raw)
    return {**record, "path": str(feature_path.relative_to(root)), "frames": len(raw),
            "final_feature_padding_samples": padding, "feature_sha256": digest(feature_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="Pinned validation.parquet and test.parquet; raw data remains outside git")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        parser.error("Use a fresh output directory and a positive worker count")
    for split, (size, sha) in SHARDS.items():
        path = args.assets/f"{split}.parquet"
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError("Pinned FLEURS shard checksum mismatch")
    args.output.mkdir(parents=True)
    seen = {}
    for official_split in SHARDS:
        split = "testing" if official_split == "test" else official_split
        root = args.output/split
        (root/"pcm").mkdir(parents=True)
        (root/"features").mkdir()
        groups, excluded, sources, audio = [], [], [], []
        speech_samples = 0

        def save_group():
            nonlocal speech_samples, sources, audio
            if not sources:
                return
            identity = f"cmn_hans_cn-{split}-{len(groups):03d}"
            path = root/"pcm"/f"{identity}.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                for pcm in audio:
                    wav.writeframes(pcm.astype("<i2").tobytes())
                    wav.writeframes(bytes(3200*2))
            groups.append({"id": identity, "split": split, "pcm_path": str(path.relative_to(root)),
                           "pcm_sha256": digest(path), "seconds": (speech_samples+3200*len(sources))/16000,
                           "speech_seconds": speech_samples/16000, "separator_seconds": .2*len(sources),
                           "sources": sources})
            speech_samples, sources, audio = 0, [], []

        parquet = pq.ParquetFile(args.assets/f"{official_split}.parquet")
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                blob = row["audio"]["bytes"]
                audio_sha = hashlib.sha256(blob).hexdigest()
                if audio_sha in seen:
                    raise ValueError("Duplicate audio within or across FLEURS partitions")
                seen[audio_sha] = split
                transcript = row["raw_transcription"]
                if ALIASES.search(transcript):
                    excluded.append({"id": row["id"], "sha256": audio_sha, "reason": "transcript_wake_alias"})
                    continue
                pcm, rate = sf.read(io.BytesIO(blob), dtype="float32")
                if rate != 16000 or pcm.ndim != 1 or len(pcm) != row["num_samples"] or not np.isfinite(pcm).all():
                    raise ValueError("Expected finite 16 kHz mono FLEURS PCM with the original sample count")
                if np.max(np.abs(pcm), initial=0) > 1:
                    raise ValueError("Unexpected float WAV amplitude; do not silently normalize")
                if speech_samples+len(pcm) > 16000*600:
                    save_group()
                # Quantize the float WAV to PCM16 once; no gain or speed change.
                quantized = np.clip(np.rint(pcm*32768), -32768, 32767).astype(np.int16)
                sources.append({"id": row["id"], "audio_sha256": audio_sha, "samples": len(pcm),
                                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest()})
                audio.append(quantized)
                speech_samples += len(pcm)
        save_group()
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(feature_worker, [(str(root), row) for row in groups]):
                records.append(record)
                print(json.dumps({"split": split, "groups": len(records), "total": len(groups)}), flush=True)
        report = {"dataset": "google/fleurs cmn_hans_cn", "license": "CC BY 4.0", "revision": REVISION,
                  "source": "https://huggingface.co/datasets/google/fleurs", "official_split": official_split,
                  "shard_sha256": SHARDS[official_split][1], "records": records, "excluded": excluded,
                  "hours": sum(r["seconds"] for r in records)/3600,
                  "speech_hours": sum(r["speech_seconds"] for r in records)/3600,
                  "speaker_identity_available": False,
                  "feature_method": "One frontend over exact PCM16 WAV, final partial hop padding only",
                  "feature_code_sha256": digest(Path(__file__).with_name("features.py")),
                  "limits": ["Public read Mandarin, not spontaneous household/TV/music or positive wake acceptance.",
                             "Official partitions retained; individual speaker IDs are unavailable in this export.",
                             "Transcript screening cannot certify acoustic absence of every wake alias.",
                             "200 ms separators are included in exposure and reported separately.",
                             "No model scores were read during preparation. Freeze candidates before testing."]}
        (root/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
        baseline = [{"path": r["pcm_path"], "sha256": r["pcm_sha256"], "samples": round(r["seconds"]*16000),
                     "voice": r["id"], "target": "none", "text": "continuous public Mandarin negative",
                     "variant": 0, "keyword_end": None, "scope": "continuous_speech_negative"} for r in records]
        (root/"baseline-manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in baseline))
        print(json.dumps({"split": split, "hours": report["hours"], "speech_hours": report["speech_hours"]}), flush=True)


if __name__ == "__main__":
    main()
