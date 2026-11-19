"""Verify the corrected matcher on the original false alert and previously selected command replays."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

import check_timed_final_marker as timed
import compare_final_marker_captures as compare
import inspect_final_marker_event as event

board = compare.board
ch = timed.ch
OUT = timed.OUT.with_name(timed.OUT.stem + "-p4.json")
WORK = board.WORK.with_name("timed-final-marker")
BUILD = WORK / "build"


def false_alert():
    old, = ch.read(event.OUT)["records"]
    result = ch.read(ch.RESULTS / "4090-c161-history-gain-jenny-v1.json")
    source, = [r["source"] for r in result["records"] if r["phase"] == "background"
               and r["source"]["pcm_sha256"] == old["source_pcm_sha256"]]
    with sf.SoundFile(source["wav_path"]) as f:
        f.seek(old["offset"])
        clip = f.read(old["samples"], dtype="int16")
    assert ch.digest(clip) == old["clip_pcm_sha256"]
    via = old["boards"]["candidate"]
    assert via["opened"] and not via["c161"]
    meta = dict(group="recorded_speech", scene=0, carrier="background", cue_at=0, command_at=len(clip),
                offset=old["offset"], file=old["source"], old_open=via["opened"], source_pcm_sha256=old["source_pcm_sha256"])
    return meta, "candidate", False, clip, np.zeros_like(clip), None, []


def requests(matcher):
    yield false_alert()
    rooms = {(r["group"],r["index"]):r for r in ch.read(timed.OUT)["records"]}
    for meta, mode, local, pcm, ref, _, c161 in board.grid_trials():
        if mode != "candidate":
            continue
        r = rooms[meta["group"], meta["scene"]]
        assert ch.digest(pcm) == r["pcm_sha256"] and c161 == r["c161"]
        if local:
            assert not r["self_opened"] and ch.digest(ref) == r["reference_sha256"]
        yield meta, mode, local, pcm, ref, None if local else r["opened"], c161
    yield from board.common_trials(matcher)


def main():
    known = ch.read(board.OUT)
    old = {(r["board"]["source"]["group"], r["board"]["source"]["scene"], r["local"]):r
           for r in known["records"] if r["mode"] == "candidate"}
    versions = known["versions"].copy()
    firmware = BUILD / "nyx_mac_listener.bin"
    model = (board.phase.listener.WORK / "model/hey_nyx_model.bin").read_bytes()
    assert ch.sha(board.phase.listener.WORK / "model/hey_nyx_model.bin") == versions["model"]
    assert firmware.read_bytes().count(model) == 1 and board.grid.gap.uplink.firmware() == versions["codec"]
    assert ch.sha(board.OUT.with_suffix(".zip")) == known["archive_sha256"]
    header_copy = WORK / "src/research/wakeword_p4/first_command_p4/mac_listener/main/final_marker.h"
    assert header_copy.read_bytes() == timed.HEADER.read_bytes()
    versions.update(previous_weak=versions["candidate"], candidate=ch.sha(firmware), host=ch.sha(timed.OUT),
        sources={p.name:ch.sha(p) for p in (timed.HEADER, timed.BG, event.OUT, board.OUT, Path(__file__),
                                             Path(board.__file__), Path(compare.__file__),
                                             Path(__file__).with_name("build_final_marker_confirmation.sh"))})
    small, info = ch.load_reader()
    assert info == versions["small"]
    model_dir = board.prior.prior.audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n,h in versions["turbo"]["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = board.grid.gap.uplink.decoder_api()
    rows, summary = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    trials = WORK / "trials"
    trials.mkdir(exist_ok=True)
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as logs, ZipFile(board.OUT.with_suffix(".zip")) as prev:
        root = Path(folder)
        matchers = (ch.binding(root/"strict.so", coded="v5"), timed.binding(root), ch.binding(root/"prefix.so", coded="aligned"))
        for i, req in enumerate(requests(matchers[1])):
            row, shard = board.run_case(trials, req, versions, (small,turbo), decoder, matchers, i, candidate_build=BUILD)
            meta, _, local = req[:3]
            key = meta["group"], meta["scene"], local
            old_row = old.get(key)
            metrics = compare.details(row)
            with ZipFile(shard) as case:
                equal = 0
                if old_row is not None:
                    assert old_row["board"]["reference_sha256"] == row["board"]["reference_sha256"]
                    equal = compare.check_equal(dict(index=old_row["index"],row=old_row), dict(index=i,row=row), prev, case)
                else:
                    assert meta["group"] == "recorded_speech" and not row["board"]["opened"] and not row["board"]["fixed"]
                for name in case.namelist():
                    if name != "row.json":
                        logs.writestr(name, case.read(name))
            groups = [meta["group"]] if meta["group"] in ("common", "recorded_speech") or meta["changed"] else []
            if meta.get("assigned"):
                groups.append("assigned")
            for group in groups:
                summary[group + (".self" if local else ".external")].update(
                    trials=1, opens=bool(metrics["opened"]), timely=metrics["timely"], small=metrics["small"], turbo=metrics["turbo"],
                    equal_packets=equal, same_open=old_row is not None and old_row["board"]["opened"] == metrics["opened"])
            row.update(previous_weak_index=None if old_row is None else old_row["index"], equal_packets=equal)
            rows.append(row)
            print(i, *key, metrics["opened"], flush=True)
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    assert len(rows) == len(old) + 1 and len({r["previous_weak_index"] for r in rows}) == len(rows)
    tmp.replace(archive)
    ch.save(OUT, dict(records=rows, summary=summary, versions=versions, archive_sha256=ch.sha(archive),
        policy="Replay the exact speech excerpt, every previous weak candidate P4 input including aligned self-play "
               "and both prespecified command sets. Decode both ASRs from the received stream. Compare actual packets "
               "against the archived weak receiver when both OPEN clocks match. Groups can overlap."))


if __name__ == "__main__":
    main()
