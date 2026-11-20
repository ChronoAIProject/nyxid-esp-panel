"""Replay identical Melo BR probe sentences after two independently muted lead-in sentences."""
from hashlib import sha256
import json
from math import lcm
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "first_command_p4/mac_listener"))
sys.path.append(str(ROOT / "experiments"))
import fixed_rearm_demand_p4 as p4

WORK = Path(".local/wakeword/nicole-context")
RESULT = ROOT / "results/4090-nicole-context.json"
SOURCE = ROOT / "results/4090-parallel-context-streams-plan-v1.json"


def main():
    source = json.loads(SOURCE.read_text())
    stream, = (s for s in source["streams"] if s["name"] == "melo-BR")
    items = {item["text_index"]: item for item in stream["items"]}
    assert [item["text_index"] for item in stream["items"][-3:]] == [34, 35, 47]
    sources = {}
    pcm = {}
    for index in (34, 35, 47, 1):
        item = items[index]
        path = Path(item["wav"])
        digest = sha256(path.read_bytes()).hexdigest()
        assert source["files"][str(path)] == digest
        audio, rate = sf.read(path, dtype="int16")
        assert rate == 16000 and audio.ndim == 1 and sha256(audio.tobytes()).hexdigest() == item["pcm_sha256"]
        sources[index] = dict(text=item["request"]["text"], path=str(path), sha256=digest)
        pcm[index] = audio

    build = json.loads((p4.WORK / "build.json").read_text())
    frozen = json.loads((ROOT / "results/4090-melo-br-nicole-original-p4.json").read_text())
    assert build == frozen["p4"]
    gap = np.zeros(source["gap_samples"], dtype="<i2")
    assert len(gap) == 8000
    prefix_samples = len(pcm[34]) + len(pcm[35]) + 2 * len(gap)
    # Move the probe by entire joint periods of the model step and the I2S read.
    period = lcm(480, 256)
    phase = prefix_samples % period
    assert (prefix_samples, period, phase) == (106095, 3840, 2415)
    prefixes = {"isolated": np.zeros(0, dtype="<i2"), "short-phase": np.zeros(phase, dtype="<i2"),
                "long-aligned": np.zeros(prefix_samples - phase, dtype="<i2"),
                "silent": np.zeros(prefix_samples, dtype="<i2")}
    for condition, audible in (("kettle", (True, False)), ("remote", (False, True)), ("both", (True, True))):
        parts = []
        for lead_index, play in zip((34, 35), audible):
            lead = pcm[lead_index]
            parts.extend([lead if play else np.zeros_like(lead), gap])
        prefixes[condition] = np.concatenate(parts)
    rows = []
    WORK.mkdir(parents=True, exist_ok=True)
    archive = RESULT.with_suffix(".zip")
    members = {}
    with TemporaryDirectory(dir=WORK) as tmp:
        with ZipFile(Path(tmp) / archive.name, "w", ZIP_DEFLATED) as output:
            for probe, index in (("nicole", 47), ("bread", 1)):
                for condition, prefix in prefixes.items():
                    label = f"{probe}-{condition}"
                    signal = np.concatenate([prefix, pcm[index]])
                    start = len(prefix)
                    path = WORK / (label + ".wav")
                    sf.write(path, signal, 16000, subtype="PCM_16")
                    row = dict(label=label, path=str(path), samples=len(signal), probe_start=start,
                               sha256=sha256(path.read_bytes()).hexdigest(), pcm_sha256=sha256(signal.tobytes()).hexdigest())
                    output.write(path, path.name)
                    members[path.name] = row["sha256"]
                    for name in ("fixed", "c161"):
                        result, log = p4.replay(name, row, WORK / name / label)
                        row[name] = result
                        member = f"{name}/{label}.log"
                        output.write(log, member)
                        members[member] = sha256(log.read_bytes()).hexdigest()
                    rows.append(row)
        with ZipFile(Path(tmp) / archive.name) as check:
            assert len(check.namelist()) == len(members) == 42
            assert all(sha256(check.read(k)).hexdigest() == v for k, v in members.items())
        (Path(tmp) / archive.name).replace(archive)
    result = dict(source_plan_sha256=sha256(SOURCE.read_bytes()).hexdigest(), sources=sources,
                  p4=build, archive_sha256=sha256(archive.read_bytes()).hexdigest(),
                  members=members, records=rows)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["label"], row["probe_start"], row["c161"]["events"], row["fixed"]["events"])


if __name__ == "__main__":
    main()
