"""Download and audit one pinned far-field channel per DiPCo development session."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import soundfile as sf

REPO = "huckiyang/DiPCo"
REVISION = "e2b29d3d0d88692c744feb15e290f7316b68014e"
# S09/S10 share participants; they must remain in the same development split.
PLAN = {"S02": "training", "S04": "training", "S05": "validation",
        "S09": "testing", "S10": "testing"}
# The pinned mirror is incomplete. Fix the first lexicographically available
# far-field channel in each session, before downloading or scoring any audio.
CHANNELS = {"S02": "U01.CH1", "S04": "U01.CH1", "S05": "U04.CH1",
            "S09": "U02.CH6", "S10": "U04.CH1"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.output
    if root.exists():
        p.error("Use a fresh directory; retain partial/failed downloads")
    root.mkdir(parents=True)
    source = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
    metadata = {}
    for name, expected in {
        "README.md": "c304173c59c6d83ffbeff20ce9e4f83dc2100cc6361d78a2e7dfb67ed839a5bc",
        "LICENSE.txt": "2d6f51b40d9b47e2b01e1503e8e16230b5745bc927ae9024f734e36fcee8e7f5",
    }.items():
        payload = urllib.request.urlopen(source + name, timeout=40).read()
        (root / name).write_bytes(payload)
        if sha(root / name) != expected:
            raise ValueError("Changed pinned license/readme")
        metadata[name] = expected
    wanted = {f"audio/dev/{session}_{CHANNELS[session]}.wav" for session in PLAN}
    entries = {}
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/audio/dev?limit=100"
    page = 0
    while url:
        with urllib.request.urlopen(url, timeout=40) as response:
            payload = response.read()
            links = response.headers.get("Link", "")
        (root / f"tree-{page:02d}.json").write_bytes(payload)
        page += 1
        for row in json.loads(payload):
            if row["path"] in wanted:
                entries[row["path"]] = row
        match = re.search(r'<([^>]+)>;\s*rel="next"', links)
        url = match.group(1) if match else None
        if url and not url.startswith(f"https://huggingface.co/api/datasets/{REPO}/"):
            raise ValueError("Unexpected pagination origin")
    if set(entries) != wanted:
        raise ValueError("Missing selected far-field channels")

    def download(session):
        remote = f"audio/dev/{session}_{CHANNELS[session]}.wav"
        entry = entries[remote]
        expected = entry["lfs"]["oid"]
        target = root / Path(remote).name
        partial = target.with_suffix(".wav.partial")
        with urllib.request.urlopen(source + remote, timeout=40) as response, partial.open("xb") as stream:
            while block := response.read(1 << 20):
                stream.write(block)
        if partial.stat().st_size != entry["size"] or sha(partial) != expected:
            raise ValueError("Audio transfer differs from pinned LFS size/hash")
        partial.rename(target)
        transcript_path = f"transcriptions/dev/{session}.json"
        payload = urllib.request.urlopen(source + transcript_path, timeout=40).read()
        transcript = root / (session + ".json")
        transcript.write_bytes(payload)
        utterances = json.loads(payload)
        speakers = sorted({r["speaker_id"] for r in utterances})
        if len(speakers) != 4 or any(r["session_id"] != session for r in utterances):
            raise ValueError("Unexpected transcript session/speaker identities")
        if any(re.search(r"\bhey\s+(nyx|nix|nicks)\b", r["words"], re.I) for r in utterances):
            raise ValueError("Literal target candidate requires review, not automatic negative labeling")
        info = sf.info(target)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames <= 0:
            raise ValueError("Unexpected source PCM format")
        pcm, _ = sf.read(target, dtype="int16")
        row = dict(path=target.name, split=PLAN[session], speaker="dipco_" + session,
                   participants=speakers, session=session, language="en", category="farfield-dialogue",
                   samples=len(pcm), seconds=len(pcm) / 16000, sha256=expected,
                   pcm_sha256=hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest(),
                   transcript_sha256=sha(transcript), source_path=remote,
                   channel=CHANNELS[session], source_partition="official dev")
        print(json.dumps(dict(session=session, samples=len(pcm), sha256=expected)), flush=True)
        return row

    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(download, PLAN))
    splits, pcm_seen = {}, set()
    for row in rows:
        for speaker in row["participants"]:
            if speaker in splits and splits[speaker] != row["split"]:
                raise ValueError("Participant crosses development splits")
            splits[speaker] = row["split"]
        if row["pcm_sha256"] in pcm_seen:
            raise ValueError("Duplicate source PCM")
        pcm_seen.add(row["pcm_sha256"])
    (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = dict(role="Audited far-field development negative corpus; not recognition evidence",
                  repo=REPO, revision=REVISION, license="CDLA-Permissive-1.0", metadata_sha256=metadata,
                  manifest_sha256=sha(root / "manifest.jsonl"), source_sha256=sha(Path(__file__)),
                  plan=PLAN, channels=CHANNELS, files=len(rows), speakers=len(splits),
                  channel_policy="First lexicographically available far-field channel per session in the incomplete pinned mirror; fixed before model scoring. Different device positions are a distribution limitation.",
                  split_hours={s: sum(r["seconds"] / 3600 for r in rows if r["split"] == s)
                               for s in sorted(set(PLAN.values()))},
                  all_audio_lfs_hashes_verified=True, exact_pcm_dedup_passed=True,
                  participant_split_disjoint=True, official_eval_downloaded=False,
                  independent_final_accessed=False, classifiers_scored=False,
                  label_limits="No literal Hey Nyx/Nix/Nicks in source transcripts; transcript absence is not a phonetic audit of every overlapping utterance. No model-based filtering.")
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
