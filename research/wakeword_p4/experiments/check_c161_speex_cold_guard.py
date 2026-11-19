"""Latch Speex adaptation at local cue onset and replay a causal reference veto on P4."""
import re
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_hold_rooms import LENGTH, OUT, RATE, STEM as PARENT, scenes, score_external
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import QEMU, boot

STEM = "4090-c161-speex-cold-guard"


def delayed(pcm, samples):
    out = np.zeros_like(pcm)
    out[samples:] = pcm[:-samples]
    return out


def extra_cases(originals):
    """Silence vs active reference, guard expiry, and a downstream speaker gain change."""
    original = {k: (m, r, s) for k, m, r, s in originals}
    for room in (0, 1):
        own, ref, self_meta = original[f"{room}-0-self-0-0"]
        # The reference is zero for this original external-only recording.
        ext, _, ext_meta = original[f"{room}-1-external_only-0-0"]
        ext = np.r_[ext[5 * RATE:], np.zeros(5 * RATE, dtype=ext.dtype)]
        for role in ("self", "simultaneous"):
            m, r, src = original[f"{room}-0-{role}-0-0"]
            at = 5 * RATE
            bounds = [v + at for v in src["known_playback"]]
            ends = src["command_end"] + at if src["command_end"] is not None else None
            yield (f"{room}-silence5-{role}", delayed(m, at), delayed(r, at),
                   dict(src, known_playback=bounds, external_start=at if ends else None,
                        command_end=ends, history="5s digital silence"))
        end = ext_meta["command_end"] - 5 * RATE
        yield (f"{room}-cold-external", ext, np.zeros_like(ref),
               dict(ext_meta, warm=False, external_start=0, command_end=end, history="external alone at boot"))
        at = 3 * RATE
        mixed = own.astype(np.int32) + delayed(ext, at)
        assert np.abs(mixed).max() <= 32767
        yield (f"{room}-cold-delayed", mixed.astype("<i2"), ref,
               dict(self_meta, role="external_after_demo", external_start=at,
                    offset=at, command_end=end+at, history="external cue 3s after cold local cue"))
        for role in ("self", "simultaneous"):
            m, r, src = original[f"{room}-1-{role}-0-0"]
            own_warm, _, _ = original[f"{room}-1-self-0-0"]
            # Double only the local acoustic echo at cue onset. Digital reference is unchanged.
            mixed = m.astype(np.int32)
            mixed[5*RATE:] += own_warm[5*RATE:]
            scale = min(1., 28000 / np.abs(mixed).max())
            yield (f"{room}-gain2-{role}", np.rint(mixed*scale).astype("<i2"),
                   np.rint(r*scale).astype("<i2"),
                   dict(src, history="5s narration then local acoustic gain x2",
                        scale=scale, external_db=-6.0206 if role == "simultaneous" else None))


def main():
    originals = list(scenes())
    parent_plan = RESULTS / f"{PARENT}-plan.json"
    assert [dict(key=k, **m) for k, _, _, m in originals] == read(parent_plan)["records"]
    parent = RESULTS / f"{PARENT}.json"
    prior = {r["key"]: r["arms"]["hold"] for r in read(parent)["records"]}
    assert sha(parent_plan) == read(parent)["plan_sha256"]
    replay(STEM, parent, originals + list(extra_cases(originals)), prior)


def replay(stem, parent, cases, prior):
    rows = []
    for key, mic, ref, src in cases:
        assert len(mic) == len(ref) == LENGTH
        rows.append(dict(src, key=key, mic_sha256=digest(mic), reference_sha256=digest(ref)))
    assert len({r["key"] for r in rows}) == len(rows)
    plan = RESULTS / f"{stem}-plan.json"
    config = dict(parent_sha256=sha(parent), records=rows)
    if plan.exists():
        assert read(plan) == config
    else:
        save(plan, config)
    root = OUT / stem
    root.mkdir(exist_ok=True)
    build = OUT / "build-cold-guard"
    report = RESULTS / f"{stem}.json"
    archive = RESULTS / f"{stem}-uart.zip"
    if report.exists() or archive.exists():
        raise FileExistsError(stem)
    reader, identity = load_reader()
    records = []
    with ZipFile(root / "uart.zip", "w", ZIP_DEFLATED) as uart:
        for key, mic, ref, src in cases:
            header = struct.pack("<II", 0x35434541, LENGTH)
            payload = header + mic.tobytes() + ref.tobytes() + struct.pack("<II", *src["known_playback"])
            log = root / f"{key}.log"
            boot(build, log, root / "flash.bin", b"SPEEX PASS", (2 << 20, payload))
            pcm, costs, heap = parse(log, LENGTH)
            text = log.read_text()
            policy = re.findall(r"^SPEEX_POLICY at=(\d+) adapted=(\d+)$", text, re.M)
            begin, end = src["known_playback"]
            assert f"SPEEX_LOCAL begin={begin} end={end}" in text
            assert [int(t) for t, _ in policy] == ([begin//160*160] if end > begin else [])
            hits = list(map(int, re.findall(r"^SPEEX_CUE (\d+)$", text, re.M)))
            blocked = list(map(int, re.findall(r"^SPEEX_BLOCKED (\d+)$", text, re.M)))
            result = dict(key=key, adapted_at_local_start=int(policy[0][1]) if policy else None,
                          hits=hits, blocked=blocked, pcm_sha256=digest(pcm), uart_sha256=sha(log),
                          median_us=float(np.median(costs)), max_us=max(costs),
                          over_10ms=sum(v > 10000 for v in costs), heap=heap)
            if key in prior:
                assert result["pcm_sha256"] == prior[key]["pcm_sha256"]
                assert sorted(hits + blocked) == prior[key]["hits"]
            result.update(score_external(reader, pcm, hits, src))
            records.append(result)
            uart.write(log, log.name)
            print(key, result["adapted_at_local_start"], hits, blocked, result.get("exact_suffix"), flush=True)
    (root / "uart.zip").replace(archive)
    save(report, dict(plan_sha256=sha(plan), elf_sha256=sha(build / "nyx_speex_echo.elf"),
                      qemu_sha256=sha(QEMU), asr=identity, uart_archive_sha256=sha(archive), records=records))


if __name__ == "__main__":
    main()
