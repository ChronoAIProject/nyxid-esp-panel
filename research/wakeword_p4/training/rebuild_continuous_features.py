"""Rebuild exact-PCM features in a fresh corpus without changing benchmark WAVs.

Legacy feature preparation padded every utterance to a complete hop. Preserve
the old artifacts and the PCM manifest used by the running Hi ESP benchmark;
only recompute features from those hash-verified, complete WAVs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from features import continuous_wav_features


def worker(job):
    record, source, output = job
    source, output = Path(source), Path(output)
    wav = source/record["pcm_path"]
    if hashlib.sha256(wav.read_bytes()).hexdigest() != record["pcm_sha256"]:
        raise ValueError("WAV identity changed")
    data, padding = continuous_wav_features(wav)
    destination = output/record["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, data)
    pcm_link = output/record["pcm_path"]
    pcm_link.parent.mkdir(parents=True, exist_ok=True)
    pcm_link.symlink_to(wav.resolve())
    return {**record, "frames": len(data), "final_feature_padding_samples": padding,
            "feature_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.corpus/"manifest.json"
    corpus = json.loads(path.read_text())
    records = []
    jobs = [(record, str(args.corpus), str(args.output)) for record in corpus["records"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(worker, jobs):
            records.append(record)
            print(json.dumps({"completed": len(records), "total": len(jobs)}), flush=True)
    # Preserve the exact bytes/hash bound to the already measured Hi ESP input.
    baseline = args.corpus/"baseline-manifest.jsonl"
    if baseline.exists():
        data = baseline.read_bytes()
        rows = list(map(json.loads, data.decode().splitlines()))
        if len(rows) != len(records):
            raise ValueError("Benchmark manifest count mismatch")
        for row, record in zip(rows, records):
            if (row["sha256"] != record["pcm_sha256"]
                    or (args.corpus/row["path"]).resolve() != (args.corpus/record["pcm_path"]).resolve()):
                raise ValueError("Benchmark PCM identity mismatch")
        (args.output/baseline.name).write_bytes(data)
    report = {**corpus, "records": records,
              "feature_method": "One frontend over exact WAV, whole-hop blocks, padding only the final incomplete 10 ms hop",
              "previous_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rebuild_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report["feature_code_sha256"] = hashlib.sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
