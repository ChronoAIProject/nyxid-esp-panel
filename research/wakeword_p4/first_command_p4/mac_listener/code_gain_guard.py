"""Replay v5 and its doubled-code WAV as external and local playback in the three missed rooms."""
import argparse
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import replay as listener
from self_guard import expected
from transition import BUILD
from check_c161_v5_code_gain import CANDIDATE, PREFIX, REPORT, WAV, playback, waves
from check_c161_v3_two_anchor_v1 import binding
from replay_c161_pcm_cues_v1 import detect


def score(work, source, mic, ref, detector, index, build=BUILD, prefix=None, qualify=None,
          wait_for_third=False):
    protocol = 0x4752594e if prefix is None else 0x5052594e
    if qualify is not None:
        protocol = 0x4452594e
    if wait_for_third:
        assert qualify is not None and prefix is not None
        protocol = 0x5452594e
    payload = struct.pack("<3I", protocol, index, len(mic)) + mic.tobytes() + ref.tobytes()
    log = work / "serial.log"
    listener.boot(build, log, work/"image.bin", b"LISTENER PASS", (2<<20, payload))
    log_text = log.read_text()
    assert listener.values(log_text, "READY") == [index, len(mic), 256, 81600]
    hits = [tuple(map(int, s.split())) for s in re.findall(r"^HITS (.+)$", log_text, re.M)]
    c161 = [t for _, t, _ in hits if t >= 0]
    fixed = [t for _, _, t in hits if t >= 0]
    assert fixed == [r["t"] for r in detect(detector, [mic])]
    refs = [r["t"] for r in detect(detector, [ref])]
    prefixes = [r["t"] for r in detect(prefix, [ref])] if prefix is not None else []
    accepted_prefixes = prefixes if qualify is None else [t for t in prefixes if qualify(ref, t)]
    tentative = sorted(set(prefixes) - set(accepted_prefixes)) if wait_for_third else []
    allows = []
    for label, events in (("C161_ALLOW", c161), ("V5_ALLOW", fixed)):
        actual = [int(s) for s in re.findall("^"+label+r" (\d+)$", log_text, re.M)]
        assert actual == expected(events, refs, ref, accepted_prefixes, tentative)
        allows.extend(actual)
    opened = [int(s) for s in re.findall(r"^OPEN (\d+) ", log_text, re.M)]
    assert opened == ([(min(allows)+255)//256*256] if allows else [])
    summary = [s for s in log_text.splitlines() if s.startswith((
        "READY ", "HITS ", "CUE_ROUTE ", "C161_ALLOW ", "V5_ALLOW ", "OPEN ", "FINISH ", "LISTENER PASS"))]
    details = dict(prefix_reference=prefixes) if prefix is not None else {}
    if qualify is not None:
        details["qualified_prefix"] = accepted_prefixes
    return dict(source=source, mic_sha256=listener.digest(mic), reference_sha256=listener.digest(ref), **details,
                c161=c161, fixed=fixed, reference_events=refs, allowed=sorted(allows), opened=opened,
                uart_sha256=listener.sha(log), uart_events=summary)


def missed_rooms():
    old, new = waves()
    original = listener.read(REPORT)["records"]
    with patch.object(playback, "TARGET", "v5"):
        for label, cue in (("v5", old), ("doubled", new)):
            for r, (src, mic) in zip(original, playback.measured(cue, gain_reference=old), strict=True):
                if src["role"] != "v5" or r["events"]:
                    continue
                assert r["source"] == src
                if label == "v5":
                    assert listener.digest(mic) == r["pcm_sha256"]
                for role in ("external", "self"):
                    ref = np.zeros_like(mic)
                    if role == "self":
                        # Source playback clock precedes the acoustic RIR delay.
                        ref[14537:14537+len(cue)] = cue
                    yield label, role, src, mic, ref


def main(out):
    historical = listener.read(listener.RESULTS/"4090-c161-mac-self-cue-guard.json")
    firmware = BUILD/"nyx_mac_listener.bin"
    assert listener.sha(firmware) == historical["firmware_sha256"]
    rows = []
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        detector = binding(work/"cue.so", coded="v5")
        for label, role, src, mic, ref in missed_rooms():
            row = score(work, src, mic, ref, detector, len(rows))
            rows.append(dict(wav=label, role=role, **row))
            print(label, src["rir"], src["noisy"], role, row["opened"], flush=True)
    assert len(rows) == 24
    listener.save(out, dict(original_report_sha256=listener.sha(REPORT), v5_sha256=listener.sha(WAV),
        doubled_sha256=listener.sha(CANDIDATE), firmware_sha256=listener.sha(firmware),
        qemu_sha256=listener.sha(listener.QEMU), records=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=listener.RESULTS/f"{PREFIX}-guard.json")
    main(p.parse_args().out)
