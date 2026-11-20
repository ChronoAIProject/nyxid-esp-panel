"""Locate the restaurant playback command on a rev1 M/R capture and compare the UART clocks."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import wave
from zipfile import ZipFile

import numpy as np
from mr_console import decode, RATE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mac_listener"))
from measure_connected_command_clock import channel, locate

# Both registered PRESTO trials use the same 16 kHz address and command starts.
ADDRESS = 21600
COMMAND = 44800
PLAYBACKS = {
    "167ca4cf055337e6ff0213c0cc46e1117b46da69f408fccee72b0d4eb2385ae6":
        ("hye-nyx-willow-six-indoor-equal-levels.zip", "PRESTO-zero-snr-nyx.wav"),
    "3f9b05a1706b17e517dfc7aa925dd37405c6c680b220acb922cc91cf06612736":
        ("hye-nyx-willow-indoor-whole-speaker-v1.zip", "PRESTO-zero-snr-nyx-whole.wav"),
}
DECISION = re.compile(
    rb"panel\.wake\.marker: decision epoch=(\d+) c161_end=(\d+) cue_end=(\d+) read_end=(\d+) "
    rb"decision_us=(\d+) start_call=(\w+) min_internal_boot=(\d+)"
)


def alignment(mic, source, origin):
    fit = locate(mic, source, origin)
    return dict(status=fit["status"], origin_capture_sample=fit.get("estimated_command_sample"),
                windows=[dict(playback_sample=origin + window["suffix_offset"], capture_sample=window["peak"],
                              correlation=window["correlation"], alternative=window["alternative"])
                         for window in fit.get("windows", [])])


def report(trial):
    capture = json.loads((trial / "capture.json").read_text())
    lines = (trial / "uart.log").read_bytes().splitlines()
    pcm, snapshot = decode(lines, capture["descriptor_elf_sha256"])
    if any(capture.get(k) != value for k, value in snapshot.items()):
        raise ValueError("Capture JSON differs from the UART snapshot")
    digest = capture["playback_sha256"]
    if digest not in PLAYBACKS:
        raise ValueError("Expected an issued PRESTO Nyx playback")
    archive, member = PLAYBACKS[digest]
    with ZipFile(HERE.parents[1] / "results" / archive) as source:
        data = source.read(member)
    if sha256(data).hexdigest() != digest:
        raise ValueError("Playback bytes differ from the recording receipt")
    with wave.open(BytesIO(data)) as wav:
        playback = channel(wav, (1,))
    mic = np.frombuffer(pcm, dtype="<i2")[::2]
    prefix = alignment(mic, playback[:COMMAND], 0)
    suffix = alignment(mic, playback, COMMAND)
    decisions = []
    for line in lines:
        match = DECISION.search(line)
        if not match or int(match[1]) != snapshot["epoch"]:
            continue
        decisions.append(dict(c161_end=int(match[2]), cue_end=int(match[3]), read_end=int(match[4]),
                              decision_us=int(match[5]), start_call=match[6].decode(),
                              min_internal_boot=int(match[7])))
    output = dict(playback=member, playback_sha256=digest, snapshot=snapshot, decisions=decisions,
                  prefix=prefix, suffix=suffix)
    if prefix["status"] != "matched" or suffix["status"] != "matched":
        return dict(output, status="inconclusive_waveform")
    boundary = suffix["origin_capture_sample"]
    playback_start = prefix["origin_capture_sample"]
    prefix_projection = playback_start + COMMAND
    output.update(estimated_command_capture_sample=boundary, prefix_projection_sample=prefix_projection,
                  estimated_address_capture_sample=playback_start + ADDRESS)
    if abs(boundary - prefix_projection) > 192 or not 0 <= boundary < snapshot["frames"]:
        return dict(output, status="inconclusive_clocks")
    if len(decisions) != 1:
        return dict(output, status="decision_missing" if not decisions else "multiple_decisions")
    event, first = decisions[0], snapshot["first_epoch_sample"]
    cue, delivered = event["cue_end"] - first, event["read_end"] - first
    if event["c161_end"] or not 0 < cue <= delivered <= snapshot["frames"]:
        return dict(output, status="decision_outside_capture_or_unexpected_route")
    if not playback_start + ADDRESS < cue <= playback_start + len(playback):
        return dict(output, status="decision_outside_address_and_command")
    if event["start_call"] != "ESP_OK":
        return dict(output, status="call_start_failed")
    return dict(output, status="matched", cue_end_capture_sample=cue, read_end_capture_sample=delivered,
                cue_lead_ms=1000 * (boundary - cue) / RATE,
                read_end_lead_ms=1000 * (boundary - delivered) / RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path, help="Output directory from mr_console.py record")
    print(json.dumps(report(parser.parse_args().trial), indent=2))
