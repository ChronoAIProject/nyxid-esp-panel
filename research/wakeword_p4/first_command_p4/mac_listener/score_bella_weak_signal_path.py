"""Locate complete-command losses along the original 32 target-gain playback paths."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import check_bella_daily_overlap as daily
import score_bella_weak_speaker_gain as scored

P, prior = scored.P, scored.prior
OUT = scored.OUT.with_name("4090-c161-bella-weak-signal-path.json")


def window(pcm, first, stop):
    return np.pad(pcm, (0, max(0, stop - len(pcm))))[first:stop]


def main():
    data = P.read(scored.OUT)
    source = {r["index"]: r for r in data["records"] if r["role"] == "nyx" and r["mode"] == "target_x2"}
    history = {r["index"]: r for r in P.read(daily.OUT)["records"] if r["role"] == "original"}
    assert len(source) == 32 and P.sha(scored.TRACE) == data["archive_sha256"]
    grid, taps = daily.rooms()
    rooms = {r[0]["file"]: r for r in grid[-6]}
    proc, model, identity = prior.load()
    assert identity == data["qwen"] and identity["device"].endswith("RTX 4090")
    turbo = prior.reader(data["turbo"])
    readings_cache, rows = {}, []
    totals = defaultdict(Counter)
    with ZipFile(scored.TRACE) as logs:
        for src, key, _, speech in daily.event_inputs("daily"):
            i = int(key)
            if i not in source:
                continue
            case, old = source[i], history[i]
            result = case["uplink"]
            assert case["timely"] and old["dry_speech_sha256"] == P.digest(speech) and old["text"] == src["text"]
            _, _, render, _, active, tracks = rooms[case["file"]]
            talk, levels = daily.tx.balanced.match_level(active, speech)
            assert levels["gain_db"] == old["speech_gain_db"]
            dry = tracks["original"].astype(np.int32) * 2
            both = dry.copy()
            both[old["speech_start"]:old["speech_end"]] += talk.astype(np.int32)
            mic, clips = render(daily.board.bella.rear_filter(both, taps))
            assert clips == 0 and P.digest(mic) == case["mic_sha256"]
            first, stop = result["first"], result["stop"]
            assert stop - first == result["samples"] and first < case["command_at"] < stop
            trace = scored.sender.archived(logs, case["label"] + "/capture.log", result["capture_uart_sha256"])
            fifo = prior.samples(trace, "PROCESSED", result["samples"])
            assert P.digest(fifo) == result["fifo_sha256"]
            # Use the same quarter-amplitude player scale as the physical PCM24 pack.
            # Quantize only once to the common int16 ASR input format.
            def for_player(pcm):
                scaled = np.rint(window(pcm, first, stop) * .25)
                assert np.max(np.abs(scaled)) < 32768
                return scaled.astype("<i2")

            views = {}
            for stage, audio in (("player_target", for_player(dry)), ("player_both", for_player(both)),
                                 ("mic", window(mic, first, stop)), ("fifo", fifo)):
                digest = P.digest(audio)
                cache_key = digest, case["command"]
                if cache_key not in readings_cache:
                    qwen = prior.evaluate(proc, model, audio, case["command"])
                    assert qwen["pcm_sha256"] == digest
                    readings_cache[cache_key] = dict(qwen=qwen, turbo=prior.recognition(turbo, audio, case["command"]))
                views[stage] = readings_cache[cache_key]
            views["sent"] = result["readings"]["immediate"]
            assert views["sent"]["qwen"]["pcm_sha256"] == result["decoded_sha256"]
            for stage, reading in views.items():
                totals[stage].update(cases=1, qwen=int(reading["qwen"]["complete"]),
                                     turbo=int(reading["turbo"]["complete_suffix"]))
            rows.append(dict(index=i, file=case["file"], command=case["command"], interruption=old["text"],
                             opened=case["opened"], first=first, stop=stop, command_at=case["command_at"],
                             mic_sha256=case["mic_sha256"], capture_uart_sha256=result["capture_uart_sha256"], views=views))
            if len(rows) % 8 == 0:
                print(len(rows), "of 32", flush=True)
    assert len(rows) == len({r["index"] for r in rows}) == 32 and len({r["file"] for r in rows}) == 8
    assert totals["sent"]["qwen"] == totals["sent"]["turbo"] == 10
    with TemporaryDirectory(dir=scored.trial.WORK) as tmp:
        output = Path(tmp) / OUT.name
        P.save(output, dict(records=rows, counts=totals, qwen=identity, turbo=data["turbo"],
                   sources={p.name:P.sha(p) for p in (Path(__file__), scored.OUT, scored.TRACE, daily.OUT, scored.trial.PLAN)},
                   policy="All 32 previously consumed Nyx target_x2 cases. Reuse each actual first and stop "
                          "sample from immediate readiness. The dry target and its mix with interruption use "
                          "one-quarter player amplitude. Then score the raw room mic window, the exact processed "
                          "PCM before Opus, and the archived decoded immediate stream. No level matching, learned "
                          "correction or latency shift between stages. Command success uses the same complete-word "
                          "suffix rules as the source. There are eight distinct room/command files with four "
                          "interruption variants each."))
        output.replace(OUT)


if __name__ == "__main__":
    main()
