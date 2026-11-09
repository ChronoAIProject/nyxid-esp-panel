"""Inspect synthesis with an independent, unprompted phoneme CTC recognizer.

This is a pronunciation diagnostic, never a source of replacement wake labels.
Only training audio is inspected. Original PCM and source manifests stay intact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="Existing diagnostic JSONL with source-relative paths; no ASR labels are adopted")
    parser.add_argument("--model", type=Path, required=True,
                        help="Pinned local phoneme model with checksum-verified identity.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh diagnostic output directory")
    if args.threads < 1:
        parser.error("--threads must be positive")
    identity = json.loads((args.model/"identity.json").read_text())
    weights = args.model/"pytorch_model.bin"
    with weights.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != identity["model_sha256"]:
            raise ValueError("Phoneme model checksum mismatch")
    rows = [json.loads(line) for line in (args.source/"manifest.jsonl").read_text().splitlines()]
    by_path = {r["path"]: r for r in rows}
    if len(by_path) != len(rows):
        raise ValueError("Duplicate source paths")
    selection = [json.loads(line) for line in args.selection.read_text().splitlines()]
    paths = [r["path"] for r in selection]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate diagnostic selections")
    selected = []
    for item in selection:
        row = by_path[item["path"]]
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or row["split"] != "training":
            raise ValueError("Only source-relative training audio is permitted")
        if item["sha256"] != row["sha256"]:
            raise ValueError("Selection/source audio identities differ")
        selected.append(row)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, local_files_only=True, weights_only=True)
    model.eval()
    vocab = json.loads((args.model/"vocab.json").read_text())
    tokens = {value: key for key, value in vocab.items()}
    blank = model.config.pad_token_id
    args.output.mkdir(parents=True)
    results = []
    with (args.output/"phonemes.jsonl").open("w") as log, torch.inference_mode():
        for number, row in enumerate(selected, 1):
            path = args.source/row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Source audio checksum mismatch: {row['path']}")
            with wave.open(str(path), "rb") as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError("Expected unchanged 16 kHz mono PCM16 source")
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)/32768
            inputs = extractor(pcm, sampling_rate=16000, return_tensors="pt")
            logits = model(**inputs).logits[0]
            # Raw greedy CTC, without a lexicon, language model or expected-text prompt.
            ids = torch.unique_consecutive(logits.argmax(dim=-1)).tolist()
            phones = [tokens[i] for i in ids if i != blank]
            result = {"path": row["path"], "sha256": row["sha256"], "expected": row["text"],
                      "source_label": row["label"], "phonemes": phones,
                      "phoneme_text": " ".join(phones), "seconds": len(pcm)/16000}
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False)+"\n")
            log.flush()
            if number % 12 == 0 or number == len(selected):
                print(json.dumps({"completed": number, "total": len(selected)}), flush=True)
    report = {"recognizer": identity, "clips": len(results), "split": "training",
              "source_manifest_sha256": hashlib.sha256((args.source/"manifest.jsonl").read_bytes()).hexdigest(),
              "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              "positive_phonemes": dict(Counter(r["phoneme_text"] for r in results if r["source_label"])),
              "negative_phonemes": dict(Counter(r["phoneme_text"] for r in results if not r["source_label"])),
              "limits": ["Learned phoneme predictions are imperfect; they do not establish human pronunciation acceptance.",
                         "No source audio, split assignment or wake label is changed.",
                         "No wake detector is evaluated and no testing audio is inspected."]}
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
