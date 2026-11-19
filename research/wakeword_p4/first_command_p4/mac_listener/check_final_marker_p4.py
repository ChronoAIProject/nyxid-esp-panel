"""Recheck every changed OPEN on P4, plus previously assigned music and room commands."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import fftconvolve

import check_final_marker_confirmation as host
import check_marker_music_commands as prior
from replay_c161_pcm_cues_v1 import detect

ch, grid, phase = host.ch, host.grid, prior.prior.p4.phase
OUT = ch.RESULTS / "4090-c161-final-marker-confirmation-p4.json"
WORK = phase.listener.WORK / "final-marker-confirmation"
BUILD = WORK / "build"


def grid_trials():
    dry = host.study.tracks()
    rows = ch.read(host.OUT)["records"]
    for group, stream in (("established", host.study.established(dry)), ("other_music", host.study.more_music(dry))):
        for i, (s, waves, _) in enumerate(stream):
            r = rows[i if group == "established" else i + 672]
            assert r["source"] == s and r["group"] == group and r["pcm_sha256"] == ch.digest(waves["v5"])
            changed = r["opened"] != r["previous_opened"]
            room, offset = divmod(i, 22)
            song, level = divmod(offset, 2)
            assigned = group == "other_music" and room == (3 * song + 1) % 28 and level == song % 2
            if not changed and not assigned:
                continue
            meta = dict(group=group, scene=i, carrier=s["command"], cue_at=s["start"], command_at=s["command_at"],
                        room=s["room"], assigned=assigned, changed=changed)
            mic = waves["v5"]
            quiet = np.zeros_like(mic)
            if changed:
                yield meta, "original", False, mic, quiet, r["previous_opened"], r["c161"]
                ref = host.study.prior.level.guard.refs(s, "v5", dry)
                assert ch.digest(ref) == r["reference_sha256"] and not r["self_opened"]
                yield meta, "candidate", True, mic, ref, None, r["c161"]
            yield meta, "candidate", False, mic, quiet, r["opened"], r["c161"]


def common_trials(matcher):
    dry = grid.fit.original()[1]
    historical = [r for r in ch.read(prior.prior.OUT)["records"] if r["group"] == "common" and r["name"] == "v5"]
    for i, (material, original) in enumerate(zip(prior.prior.audio.room.room_inputs(), historical, strict=True)):
        s, _, kind, h, bg, command = material
        end = s["end"]
        start = end - len(dry)
        wet = fftconvolve(dry.astype(float), h)
        mix = bg + command
        mix[start:start + len(wet)] += wet
        pcm = grid.room.quantize(mix * s["gain"])
        board = original["board"]
        assert ch.digest(pcm) == board["mic_sha256"]
        detections = host.joined(board["c161"], detect(matcher, [pcm]))
        opened = grid.gap.study.voices.timing.block(detections[0]) if detections else None
        meta = dict(group="common", scene=i, carrier=s["carrier"], cue_at=start, command_at=end, kind=kind,
                    room=s["room"], assigned=False, changed=board["opened"] != ([] if opened is None else [opened]))
        yield meta, "candidate", False, pcm, np.zeros_like(pcm), opened, board["c161"]


def run_case(work, request, versions, readers, decoder, matchers, i):
    meta, mode, local, pcm, ref, opened, c161 = request
    key = dict(source=meta, mode=mode, local=local, mic=ch.digest(pcm), ref=ch.digest(ref), opened=opened, versions=versions)
    checksum = sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    archive = work / f"{i:03}-{checksum[:16]}.zip"
    if archive.exists():
        with ZipFile(archive) as logs:
            row = json.loads(logs.read("row.json"))
            assert row["key_sha256"] == checksum and logs.testzip() is None
        return row, archive
    strict, candidate, prefix = matchers
    mic_matcher = strict if mode == "original" else candidate
    build = phase.handoff.test.BUILD if mode == "original" else BUILD
    tmp = archive.with_suffix(".tmp")
    with ZipFile(tmp, "w", ZIP_DEFLATED) as logs:
        for attempt in range(3):
            try:
                board = phase.handoff.probe.score(work, meta, pcm, ref, mic_matcher, i, build=build, prefix=prefix,
                        qualify=phase.handoff.probe.qualify, wait_for_third=True, frame_activity=True,
                        qualify_capture_frame=True, reference_hits=lambda x: [e["t"] for e in detect(strict, [x])])
                break
            except (AssertionError, TimeoutError):
                logs.write(work / "serial.log", f"retry-{i:03}-{attempt}.txt")
                if attempt == 2:
                    raise
        assert board["opened"] == ([] if opened is None else [opened]) and board["c161"] == c161
        logs.writestr(f"events/{i:03}.txt", "\n".join(board["uart_events"]) + "\n")
        row = dict(index=i, mode=mode, local=local, key_sha256=checksum, board=board)
        if board["opened"]:
            assert not local
            cap = phase.handoff.bench.fifo(work, logs, i, pcm, ref, board, readers[0], decoder, True, True)
            wave, count = prior.prior.audio.decoded(logs, f"{i:02}-opus.log", decoder, i, cap["samples"], 0, cap["pcm_sha256"])
            assert count == cap["packets"]
            cap["turbo"] = grid.gap.uplink.recognition(readers[1], wave, meta["carrier"])
            row["uplink"] = cap
        logs.writestr("row.json", json.dumps(row))
    tmp.replace(archive)
    return row, archive


def main():
    small, identity = ch.load_reader()
    previous = ch.read(prior.prior.OUT)
    info = previous["turbo"]
    model_dir = prior.prior.audio.prior.context.prior.audit.TURBO
    assert identity == previous["small"] and all(ch.sha(model_dir / n) == h for n, h in info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = grid.gap.uplink.decoder_api()
    binaries = [b / "nyx_mac_listener.bin" for b in (phase.handoff.test.BUILD, BUILD)]
    embedded = (phase.listener.WORK / "model/hey_nyx_model.bin").read_bytes()
    assert all(b.read_bytes().count(embedded) == 1 for b in binaries) and ch.sha(binaries[0]) == previous["firmware_sha256"]
    scripts = [Path(__file__), Path(host.__file__), Path(__file__).with_name("build_final_marker_confirmation.sh")]
    versions = dict(original=ch.sha(binaries[0]), candidate=ch.sha(binaries[1]), model=sha256(embedded).hexdigest(),
                    host=ch.sha(host.OUT), codec=grid.gap.uplink.firmware(), small=identity, turbo=info,
                    sources={p.name:ch.sha(p) for p in scripts})
    work = WORK / "trials"
    work.mkdir(exist_ok=True)
    from tempfile import TemporaryDirectory
    from itertools import chain
    rows, summary = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as uart:
        folder = Path(tmp)
        strict, prefix = ch.binding(folder / "strict.so", coded="v5"), ch.binding(folder / "prefix.so", coded="aligned")
        candidate = host.binding(folder)
        for i, request in enumerate(chain(grid_trials(), common_trials(candidate))):
            row, shard = run_case(work, request, versions, (small, turbo), decoder, (strict, candidate, prefix), i)
            meta, mode, local = request[:3]
            cap = row.get("uplink")
            opened = row["board"]["opened"]
            groups = [meta["group"] + "_changed"] if meta["changed"] else []
            if meta["assigned"]: groups.append("assigned")
            if meta["group"] == "common": groups.append("common")
            for g in groups:
                summary[f"{g}.{mode}.{'self' if local else 'external'}"].update(
                    trials=1, opens=bool(opened), timely=bool(opened and meta["cue_at"] <= opened[0] < meta["command_at"]),
                    small=bool(cap and cap["recognition"]["complete_suffix"]), turbo=bool(cap and cap["turbo"]["complete_suffix"]))
            with ZipFile(shard) as logs:
                for name in logs.namelist():
                    if name != "row.json":
                        uart.writestr(name, logs.read(name))
            rows.append(row)
            print(i, meta["group"], meta["scene"], mode, local, opened, flush=True)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
    assert summary["assigned.candidate.external"]["trials"] == 11 and summary["common.candidate.external"]["trials"] == 16
    for g, n in (("established", 24), ("other_music", 20)):
        for route in ("original.external", "candidate.external", "candidate.self"):
            assert summary[g + "_changed." + route]["trials"] == n
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, summary=summary, versions=versions, archive_sha256=ch.sha(archive),
                      policy="Original v5 dry PCM. Replay all grid cases with changed OPEN on both P4 listeners "
                      "and run aligned self-play on the candidate. Also run the candidate on the same prespecified "
                      "eleven cross-music and sixteen common command cases. Zero-reference external captures are "
                      "transcribed using all received packets. Changed cases are deliberately selected, not a population rate."))


if __name__ == "__main__":
    main()
