"""Compare two unprompted ASR models on the exact selected-channel P4 Opus audio."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from faster_whisper import WhisperModel

import attribute_first_command_audio as audit
import replay_room_spoken_selection as replay

ch, gap = replay.ch, replay.gap
OUT = ch.RESULTS / (replay.selection.STEM + "-readers.json")


def transcribe(report, archive, cases, turbo, decoder):
    assert ch.sha(archive) == report["archive_sha256"]
    results = {}
    with ZipFile(archive) as logs:
        for i, key, cap, command in cases:
            data = logs.read(key).decode().replace("\r\n", "\n")
            packets = gap.uplink.packets(data, i, cap["emitted"]-cap["first"], 0)
            pcm = gap.uplink.decode(decoder, packets)
            stored = cap["lanes"]["selected"]
            assert len(packets) == stored["packets"] and ch.digest(pcm) == stored["decoded_sha256"]
            large = gap.uplink.recognition(turbo, pcm, command)
            results[i] = dict(decoded_sha256=ch.digest(pcm), command=command, small=stored["asr"], turbo=large)
    return results


def main():
    report = ch.read(replay.OUT)
    fixed = ch.read(replay.room.parent.OUT)
    info = ch.read(ch.RESULTS/(audit.STEM+".json"))["turbo"]
    assert all(ch.sha(audit.TURBO/n) == h for n,h in info["files"].items())
    assert report["asr"] == fixed["asr"]
    turbo = WhisperModel(str(audit.TURBO), device="cuda", compute_type="float16", num_workers=1, cpu_threads=1)
    decoder = gap.uplink.decoder_api()
    cases = [(c["id"], f'opus/{c["id"]:03}-select.log', c, report["records"][c["scene"]]["command"])
             for c in report["captures"]]
    natural = transcribe(report, replay.OUT.with_suffix(".zip"), cases, turbo, decoder)
    cases = []
    for r in fixed["records"]:
        route = r["routes"]["fixed_v5"]
        i = route["capture_id"]
        cases.append((i, f"opus/{i:03}-rooms.log", route["capture"], r["command"]))
    coded = transcribe(fixed, replay.room.parent.OUT.with_suffix(".zip"), cases, turbo, decoder)
    fit = ch.read(replay.selection.FIT)
    totals = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    def count(room, mode, detector, data, timely):
        total = totals[room][mode][detector]
        total.update(trials=1, opened=data is not None)
        if data:
            for name in ("small","turbo"):
                val = data[name]["complete_suffix"]
                total.update({name+"_complete":val, name+"_timely_complete":timely and val})
    for r, old in zip(report["records"], fixed["records"], strict=True):
        assert (r["silence_sha256"], r["command_at"]) == (old["silence_sha256"], old["command_at"])
        for choices in fit["choices"].values():
            for mode in ("default", "selected"):
                entry = r["cues"][str(choices[mode]["target"])]
                for det, route in entry["routes"].items():
                    data = natural[route["capture"]] if route["hits"] else None
                    count(r["room"], mode, det, data, route.get("timely",False))
        v5 = old["routes"]["fixed_v5"]
        count(r["room"], "fixed_v5", "combined", coded[v5["capture_id"]], v5["opened"] < r["command_at"])
    ch.save(OUT, dict(natural=natural, fixed_v5=coded, summary=totals, small_files=report["asr"], turbo=info,
                      sources={p.name:ch.sha(p) for p in (replay.OUT, replay.room.parent.OUT, replay.selection.FIT,
                               replay.OUT.with_suffix(".zip"), replay.room.parent.OUT.with_suffix(".zip"), Path(__file__))},
                      policy="Same decoded selected-channel audio and all samples including pre-roll. Compare "
                      "archived small.en and independently run large-v3-turbo, unprompted English, temperature "
                      "zero, beam three. Unopened trials count against complete-command totals. Each voice uses "
                      "the same 16 command/room cells; the v5 reference is counted on its 16 distinct cases."))


if __name__ == "__main__":
    main()
