"""Rebuild the Joe score and matched command timing review from committed inputs."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile

import numpy as np

from uint8_event_policy_v1 import event_steps


RESULTS = Path(__file__).resolve().parents[1] / "results"


def read(name):
    return json.loads((RESULTS / f"4090-{name}-v1.json").read_bytes())


def joe_scores():
    runtime = read("c161-binary40-parallel-runtime-original")
    ambient = read("c161-ambient-history-joe")
    pack = RESULTS / "4090-c161-binary40-parallel-runtime-evidence-v1.tar.gz"
    alt_path = RESULTS / "4090-c161-ambient-history-joe-scores-v1.npz"
    assert hashlib.sha256(alt_path.read_bytes()).hexdigest() == ambient["scores_sha256"]
    output = []
    with tarfile.open(pack) as archive, np.load(alt_path, allow_pickle=False) as alt:
        payload = archive.extractfile("joe/result.json").read()
        assert hashlib.sha256(payload).hexdigest() == runtime["cohorts"]["joe"]["sha256"] == ambient["source_sha256"]
        for row, other in zip(json.loads(payload)["records"], ambient["records"], strict=True):
            source = row["source"]
            if source["target"] != "hey_nyx":
                continue
            index = row["index"]
            assert row["phase"] == other["phase"] == "synthetic" and index == other["index"]
            trace = archive.extractfile(f"joe/synthetic-{index:06}.npz").read()
            assert hashlib.sha256(trace).hexdigest() == row["trace"]["sha256"] == other["trace_sha256"]
            result = {k: source[k] for k in ("id", "source_id", "source_sha256", "text", "length_scale", "draw", "variant", "lead_samples")}
            with np.load(io.BytesIO(trace), allow_pickle=False) as scores:
                clock = scores["relative_block_return"]
                for model, values in (("c161", scores["c161_uint8"]), ("binary40", scores["binary40_uint8"]),
                                      ("ambient", alt[f"synthetic_{index}"])):
                    assert len(values) == len(clock) and values.dtype == np.uint8
                    steps = event_steps(values, 235 / 256)
                    events = [int(clock[i - 1]) for i in steps]
                    original = other["ambient"] if model == "ambient" else row["events"][model]
                    assert events == [e["relative_sample"] for e in original]
                    sums = np.convolve(values.astype(int), np.ones(5, dtype=int), "valid")
                    after_prime = np.flatnonzero(clock[4:] > 0)
                    peak = after_prime[np.argmax(sums[after_prime])]
                    result[model] = dict(max5=int(sums[peak]), peak_sample=int(clock[peak + 4]), event_samples=events)
            output.append(result)
    assert len(output) == 72 and len({r["source_sha256"] for r in output}) == 24
    return output


def matched_commands():
    sources = read("c161-willow-matched-scenes-sources")
    report = read("c161-willow-matched-scenes-review")
    meta = {(s["voice"], s["scene"], s["role"]): s for s in sources["rows"]}
    rows = []
    with ZipFile(RESULTS / "4090-c161-matched-command-alignments-v1.zip") as archive:
        for row in report["records"]:
            voice, scene, role = row["voice"], row["scene"], row["target"]
            ref = meta[voice, scene, role]
            data = archive.read(f"{voice}-{scene}-{role}.npz")
            assert hashlib.sha256(data).hexdigest() == ref["alignment_sha256"]
            with np.load(io.BytesIO(data), allow_pickle=False) as tensors:
                attn = tensors["attention"][0, 0]
                assert np.all(tensors["mask"] == 1) and np.all((attn == 0) | (attn == 1))
                assert np.all(attn.sum(axis=1) == 1)
                columns = attn.argmax(axis=1)
            info = ref["phones"]
            assert attn.shape[1] == len(info["input_phone_ids"]) == 2 * len(info["phones"]) + 1
            assert np.all(np.diff(columns) >= 0) and info["phones"].count(",") == 1
            phone = info["phones"].index(",") + 1
            column = 2 * phone + 1
            assert info["input_phone_ids"][column - 1] == 0 and info["input_phone_ids"][column] != 0
            frame = int(np.flatnonzero(columns == column)[0])
            # First 16 kHz sample at/after the first lexical phone's native frame.
            sample = row["target_start"] + (frame * 512 * 16000 + 44099) // 44100
            assert row["target_start"] <= sample < row["end"]
            detector = "pair" if role == "nyx" else "willow"
            events = row["events"][detector]
            rows.append(dict(voice=voice, scene=scene, target=role, command=row["command"],
                             phone=info["phones"][phone], native_frame=frame,
                             command_phone_sample=sample, event_samples=events,
                             event_minus_command_samples=[t - sample for t in events]))
    assert len(rows) == 24
    return rows


def training_sources():
    old_asr = {r["sha256"]: r for r in read("chatterbox-training-asr-spelling-diagnostic")["records"]}
    output = []
    for cohort in ("training", "bare-training"):
        admission = read(f"chatterbox-{cohort}-admission")
        for status in ("admitted", "quarantined"):
            for row in admission[status]:
                if row["request"]["requested_label"] != 1:
                    continue
                sha = row["wav"]["sha256"]
                asr = row["asr"] if "asr" in row else old_asr[sha]
                assert asr["sha256"] == sha and asr["expected"] == row["request"]["text"]
                output.append(dict(cohort=cohort, status=status, index=row["index"], sha256=sha,
                                   transcript=asr["transcript"], contains_next=bool(re.search(r"\bnext\b", asr["transcript"], re.I))))
    return output


if __name__ == "__main__":
    print(json.dumps(dict(joe=joe_scores(), matched=matched_commands(), training=training_sources()), indent=2))
