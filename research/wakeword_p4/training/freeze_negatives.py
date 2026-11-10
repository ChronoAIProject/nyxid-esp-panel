"""Freeze hash-checked LibriSpeech test negatives without reading any model.

Builds speaker-continuous MicroFrontend inputs. No scores, model selection or
threshold tuning are performed here. Official test corpora stay outside the
training/validation stream index.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess

from features import human_worker

ARCHIVES = {
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}


def digest(path, method="sha256"):
    h = hashlib.new(method)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--training-human", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Use a fresh output directory to preserve the frozen set")
    previous_ids = {int(r["speaker"]) for r in json.loads(a.training_human.read_text())}
    records, provenance, identities = [], [], set()
    for name, expected in ARCHIVES.items():
        archive = a.work/f"{name}.tar.gz"
        if digest(archive, "md5") != expected:
            raise ValueError(f"Official checksum mismatch: {archive}")
        corpus = a.work/"LibriSpeech"/name
        if corpus.exists():
            raise ValueError(f"Use an unextracted corpus directory: {corpus}")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(a.work)], check=True)
        transcript_hashes = []
        # Candidate literal wake phrases in reference transcripts must be
        # reviewed rather than silently counted as false positives or removed.
        for path in sorted(corpus.rglob("*.trans.txt")):
            text = path.read_text()
            if re.search(r"\bhey\s+(?:nicks|nix|nyx|knicks)\b", text, re.I):
                raise ValueError(f"Possible true wake in negative corpus: {path}")
            transcript_hashes.append((str(path.relative_to(corpus)), digest(path)))
        speakers = sorted(int(path.name) for path in corpus.iterdir() if path.is_dir())
        if previous_ids.intersection(speakers) or identities.intersection(speakers):
            raise ValueError("Human speaker ID crosses development/holdout corpora")
        identities.update(speakers)
        root = a.output/name
        jobs = [(str(corpus), str(root), "testing", speaker) for speaker in speakers]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for row in pool.map(human_worker, jobs):
                row["path"] = str(Path(name)/row["path"])
                row["corpus"] = name
                row["feature_sha256"] = digest(a.output/row["path"])
                records.append(row)
        provenance.append({"corpus": name, "official_md5": expected,
                           "archive_sha256": digest(archive),
                           "url": f"https://www.openslr.org/resources/12/{name}.tar.gz",
                           "transcript_index_sha256": hashlib.sha256(json.dumps(transcript_hashes).encode()).hexdigest(),
                           "license": "CC-BY-4.0", "speaker_ids": speakers})
    manifest = {"records": records, "provenance": provenance,
                "source_hours": sum(r["source_seconds"] for r in records)/3600,
                "development_speaker_disjoint": True,
                "scores_computed": False,
                "limits": "Negatives only; no independent positive holdout or Hi ESP measurement. Each speaker's utterances are concatenated with 200 ms silence; source exposure excludes that silence. Reference-text screen is not a complete phonetic audit. Cross-corpus physical speaker independence is inferred from LibriSpeech IDs, not cross-engine voice presets. Poisson bounds apply only to measured audio distribution."}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"speakers": len(records), "source_hours": manifest["source_hours"],
                      "manifest_sha256": digest(a.output/"manifest.json"), "scores_computed": False}), flush=True)


if __name__ == "__main__":
    main()
