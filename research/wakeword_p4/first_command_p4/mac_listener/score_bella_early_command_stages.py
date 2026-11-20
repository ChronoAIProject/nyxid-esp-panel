"""Trace the same 15 weak commands across the player, microphone, FIFO and receive stream."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import package_bella_early_rate_audio as playback
import score_bella_early_marker as weak
import score_bella_received_word_times as received
import score_bella_weak_signal_path as old

P = weak.P
OUT = weak.OUT.with_name("4090-bella-early-command-stages.json")
PROTOCOL = weak.early.ROOT / "experiments/bella_early_command_stages_protocol.md"


def main():
    data = P.read(weak.OUT)
    archive = weak.OUT.with_suffix(".zip")
    pack_meta = P.read(playback.OUT)
    assert P.sha(archive) == data["archive_sha256"]
    assert P.sha(playback.PACK) == pack_meta["archive_sha256"]
    historical = {r["index"]: r for r in P.read(old.daily.OUT)["records"] if r["role"] == "original"}
    rooms, _ = old.daily.rooms()
    tracks = {r[0]["file"]: r[-1]["absent"] for r in rooms[-6]}
    plan = {r["label"]: r for r in P.read(weak.early.PLAN)["records"]}
    speeds = dict(playback.study.SPEEDS)
    cue, rate = sf.read(weak.early.WAV, dtype="int16")
    assert rate == 16000 and P.sha(weak.early.WAV) == data["cue_sha256"]
    proc, model, identity = old.prior.load()
    assert identity == data["qwen"] and "4090" in identity["device"]
    turbo = old.prior.reader(data["turbo"])
    decoder = old.prior.decoder_api()
    counts, rows = defaultdict(Counter), []
    with ZipFile(playback.PACK) as sources, ZipFile(archive) as logs, TemporaryDirectory() as work:
        manifest = json.loads(sources.read("manifest.json"))
        players = {r["path"]: r for r in manifest["records"]}
        for case in data["records"]:
            scene, tag, label = case["scene"], case["rate"], case["label"]
            name = f"{scene:03}/{tag}/early-nyx.wav"
            raw = sources.read(name)
            assert playback.early.playback.wakes.digest(raw) == pack_meta["files"][name] == players[name]["sha256"]
            external, sr = sf.read(BytesIO(raw), dtype="int32")
            assert sr == 16000 and np.all(external % 16384 == 0)
            source = historical[scene]
            target = tracks[source["file"]].astype(np.int32).copy()
            end = source["command_at"]
            target[end-len(cue):end] = cue
            speed = speeds[tag]
            if speed != 1:
                target = np.rint(resample_poly(target.astype(float), speed.denominator, speed.numerator))
            assert len(target) == len(external) and round(end / speed) == case["command_at"]
            mic, sr = sf.read(plan[label]["path"], dtype="int16")
            assert sr == 16000 and P.digest(mic) == players[name]["mic_sha256"] == case["pcm_sha256"]
            trace_name = label + "/capture.log"
            trace = weak.scoring.sender.archived(logs, trace_name, data["files"][trace_name])
            fifo = old.prior.samples(trace, "PROCESSED", case["samples"])
            assert P.digest(fifo) == case["fifo_sha256"]
            first, stop = case["first"], case["stop"]
            assert first < case["command_at"] < stop and stop - first == len(fifo)
            views = {}
            for stage, full in (("player_target", target * .25), ("player_mix", external / 65536),
                                ("mic", mic)):
                sample = np.rint(old.window(full, first, stop))
                assert sample.min() >= -32768 and sample.max() <= 32767
                views[stage] = sample.astype("<i2")
            views["fifo"] = fifo
            scores = {}
            for stage, pcm in views.items():
                qwen = old.prior.evaluate(proc, model, pcm, case["command"])
                whisper = old.prior.recognition(turbo, pcm, case["command"])
                scores[stage] = dict(qwen=qwen, turbo=whisper)
            row = dict(case, uart=data["files"])
            for condition, pcm, _ in received.read_sends(Path(work), logs, row, decoder):
                if condition == "immediate":
                    scores["received"] = case["readings"][condition]
                    assert P.digest(pcm) == scores["received"]["qwen"]["pcm_sha256"]
            for stage, scoreset in scores.items():
                counts[stage].update(cases=1, qwen=int(scoreset["qwen"]["complete"]),
                                     turbo=int(scoreset["turbo"]["complete_suffix"]))
            rows.append(dict(label=label, scene=scene, rate=tag, command=case["command"], first=first, stop=stop,
                             command_at=case["command_at"], player_file=name, capture_uart_sha256=data["files"][trace_name],
                             readings=scores))
    assert len(rows) == len({r["label"] for r in rows}) == 15
    inputs = (Path(__file__), PROTOCOL, weak.OUT, weak.early.PLAN, weak.early.WAV, playback.OUT, old.daily.OUT)
    P.save(OUT, dict(records=rows, counts=counts, qwen=identity, turbo=data["turbo"],
                     command_archive_sha256=data["archive_sha256"], player_archive_sha256=pack_meta["archive_sha256"],
                     sources={str(p):P.sha(p) for p in inputs}))


if __name__ == "__main__":
    main()
