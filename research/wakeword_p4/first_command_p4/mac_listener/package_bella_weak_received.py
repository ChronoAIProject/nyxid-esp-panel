"""Export the paired received utterances for the three weak Mac playback scenes."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import package_weak_gain_room_audio as room
import send_bella_weak_raw_mic as trial

P, scored = trial.P, trial.scored
NAME = "c161-bella-weak-received-pairs.zip"
OUT = scored.OUT.with_name("4090-c161-bella-weak-received-pairs.json")


def main():
    raw, processed = P.read(trial.OUT), P.read(scored.OUT)
    assert P.sha(trial.TRACE) == raw["archive_sha256"] and P.sha(scored.TRACE) == processed["archive_sha256"]
    captures = {r["index"]: r for r in raw["records"]}
    decoder = trial.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == raw["decoder"]
    rows = []
    with TemporaryDirectory(dir=scored.trial.WORK) as tmp, ZipFile(trial.TRACE) as raw_logs, ZipFile(scored.TRACE) as baseline_logs:
        work = Path(tmp)
        with ZipFile(work / NAME, "w") as pack:
            room.daily.matched.member(pack, "music-source-LICENSE.txt", room.LICENSE.read_bytes())
            room.daily.matched.member(pack, "music-attribution.txt", (room.CREDIT + "\n").encode())
            for number, r in enumerate(processed["records"]):
                if r["index"] not in (77, 79, 90) or (r["mode"], r["role"]) != ("target_x2", "nyx"):
                    continue
                product = r["uplink"]
                case = captures[r["index"]]
                idx = str(r["index"]).zfill(3)
                for route, capture, logs, label, opus_hash in (
                    ("processed", product, baseline_logs, r["label"], product["opus_uart_sha256"]),
                    ("raw", case, raw_logs, idx, case["codec"]["opus_uart_sha256"]),
                ):
                    count = product["samples"]
                    text = scored.sender.archived(logs, label + "/opus.log", opus_hash)
                    cold = trial.prior.packets(text, number, count, 0)
                    sender_text = scored.sender.archived(logs, label + "/sender.log", capture["sender_uart_sha256"])
                    log = work / "sender.log"
                    log.write_text(sender_text)
                    sends = scored.sender.compare(log, number, product["opened"], product["first"], count, cold)
                    for condition, (packets, timing) in zip(("immediate", "waited"), sends, strict=True):
                        pcm = trial.prior.decode(decoder, packets)
                        observed = capture["readings"][condition]
                        assert P.digest(pcm) == observed["qwen"]["pcm_sha256"]
                        name = f"{idx}/{route}-{condition}.wav"
                        wav = room.wav(pcm, "PCM_16")
                        room.daily.matched.member(pack, name, wav)
                        rows.append(dict(file=name, bytes=len(wav), sha256=sha256(wav).hexdigest(),
                                         pcm_sha256=P.digest(pcm), samples=len(pcm), command=r["command"],
                                         opened=r["opened"], first_mic_sample=product["first"],
                                         first_captured_pts_ms=timing["first_captured_pts"],
                                         qwen=observed["qwen"]["text"], turbo=observed["turbo"]["text"]))
            manifest = dict(rate=16000, format="mono PCM16", records=rows, attribution=room.CREDIT,
                            sources={p.name: P.sha(p) for p in (Path(__file__), scored.OUT, scored.TRACE, trial.OUT, trial.TRACE)},
                            playback="Simulated peer output with original decoded amplitudes, all leading samples "
                                     "and terminal padding. Use as a listening or digital receive control. The "
                                     "physical sources for board testing are in the external speaker and room packs.")
            room.daily.matched.member(pack, "manifest.json", json.dumps(manifest, indent=2) + "\n")
        with ZipFile(work / NAME) as pack:
            assert len(rows) == 12 and len(pack.namelist()) == 15 and pack.testzip() is None
        P.save(work / OUT.name, dict(archive=NAME, bytes=(work / NAME).stat().st_size, archive_sha256=P.sha(work / NAME), **manifest))
        (work / NAME).replace(scored.trial.WORK / NAME)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
