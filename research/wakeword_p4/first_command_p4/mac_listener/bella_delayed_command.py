"""Shift only the target command in the three released overlapping-speech scenes."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import bella_reference_barge as receiver
import score_bella_early_command_stages as stages

P, weak, pack = stages.P, stages.weak, stages.playback
daily, scoring = stages.old.daily, weak.scoring
WORK = Path(".local/wakeword/bella-delayed-command")
OUT = weak.OUT.with_name("4090-bella-delayed-command.json")
PROTOCOL = weak.early.ROOT / "experiments/bella_delayed_command_protocol.md"
GAP = 10240
PLAY = WORK / "hye-nyx-640ms-command-pairs-v1.zip"
TRACE = WORK / "4090-bella-delayed-command-uart.zip"


def rate_convert(dry, speed):
    if speed == 1:
        return dry
    return np.rint(resample_poly(dry.astype(float), speed.denominator, speed.numerator)).astype(np.int32)


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    base = P.read(weak.OUT)
    plan = {r["label"]: r for r in P.read(weak.early.PLAN)["records"]}
    published = P.read(pack.OUT)
    assert P.sha(pack.PACK) == published["archive_sha256"]
    history = {r["index"]: r for r in P.read(daily.OUT)["records"] if r["role"] == "original"}
    rooms, taps = daily.rooms()
    by_file = {r[0]["file"]: r for r in rooms[-6]}
    speeds = dict(pack.study.SPEEDS)
    previous = P.read(receiver.STEM.with_name(receiver.STEM.name + "-historical.json"))
    old = {r["label"]:r for r in previous["records"]}
    flash_path = receiver.WORK / "guest.flash"
    assert P.sha(flash_path) == previous["build"]["flash_sha256"]
    flash = bytearray(flash_path.read_bytes())
    fixed = receiver.test.fixed
    assert P.sha(fixed.QEMU) == previous["build"]["qemu_sha256"]
    prior = scoring.prior
    for image in (scoring.clock.BUILD / "nyx_mac_listener.bin", prior.CODEC / "nyx_first_command.bin", scoring.sender.BUILD / "board.bin"):
        assert P.sha(image) == base["sources"][str(image)]
    proc, model, qid = prior.load()
    assert qid == base["qwen"] and "4090" in qid["device"]
    decoder = prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == base["decoder"]
    readers = proc, model, prior.reader(base["turbo"]), decoder
    rows, pairs, counts = [], {}, defaultdict(Counter)
    with TemporaryDirectory(dir=WORK) as temp, ZipFile(pack.PACK) as sources:
        work = Path(temp)
        scoring.clock.reference.WORK = work
        with ZipFile(work / TRACE.name, "w", ZIP_DEFLATED) as evidence, ZipFile(work / PLAY.name, "w", ZIP_DEFLATED) as playback:
            for i, case in enumerate(base["records"]):
                scene, label, tag = case["scene"], case["label"], case["rate"]
                r, src = plan[label], history[scene]
                _, _, render, ref, _, tracks = by_file[src["file"]]
                target = tracks["absent"].astype(np.int32)
                assert not np.any(target[:src["command_at"]]) and not np.any(target[-GAP:])
                name = f"{scene:03}/nominal/early-nyx.wav"
                raw = sources.read(name)
                assert fixed.digest(raw) == published["files"][name]
                high, sr = sf.read(BytesIO(raw), dtype="int32")
                assert sr == 16000 and high.shape == target.shape and np.all(high % 16384 == 0)
                original = high // 16384
                delayed = original - target
                delayed[GAP:] += target[:-GAP]
                assert np.array_equal(delayed[:src["command_at"]], original[:src["command_at"]])
                speed = speeds[tag]
                for variant, dry in (("original", original), ("delayed", delayed)):
                    played = rate_convert(dry, speed)
                    rear = daily.board.bella.rear_filter(played, taps)
                    padded = np.pad(rear, (0, max(0, len(dry)-len(rear))))[:len(dry)]
                    mic, clips = render(padded)
                    assert not clips and P.digest(ref) == case["reference_sha256"]
                    if variant == "original":
                        assert P.digest(mic) == case["pcm_sha256"]
                    if tag == "nominal":
                        assert np.max(np.abs(played)) < 131072
                        wav = pack.early.playback.playback.sources.wav(played.astype(np.int32)*16384, "PCM_24")
                        if variant == "original":
                            assert wav == raw
                        member = f"{scene:03}/{variant}.wav"
                        playback.writestr(member, wav)
                        pairs[member] = dict(sha256=fixed.digest(wav), bytes=len(wav), mic_sha256=P.digest(mic),
                                             command=case["command"], command_at=src["command_at"]+(GAP if variant=="delayed" else 0))
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                epoch, log, faults = fixed.boot(work, flash, work, block, 1)
                evidence.write(log, label + "/wake.log")
                for fault in faults:
                    evidence.write(fault, label + "/" + fault.name)
                events = epoch[0]["events"]
                before = old["rate/"+label]
                assert before["candidate"]["events"]["O"] == [case["opened"]]
                opened = next(iter(events["O"]), None)
                end = round((src["command_at"] + GAP)/speed)
                timely = opened is not None and r["start"] <= opened < end
                wav_path = work/(label+".wav")
                sf.write(wav_path, mic, 16000, subtype="PCM_16")
                row = dict(label=label, scene=scene, rate=tag, original_open=case["opened"], events=events, timely=timely,
                           command=case["command"], original_command_at=case["command_at"], command_at=end, input_sha256=fixed.digest(block),
                           original_mic_sha256=case["pcm_sha256"], mic_sha256=P.digest(mic), ref_sha256=P.digest(ref))
                if opened is not None:
                    record = dict(label=label, command=case["command"], path=str(wav_path), reference_path=r["reference_path"],
                                  pcm_sha256=row["mic_sha256"], reference_sha256=row["ref_sha256"])
                    observed, shard = scoring.score_case(record, i, opened, readers, work, work)
                    with ZipFile(shard) as z:
                        for member in z.namelist():
                            if member.endswith(".log"):
                                evidence.writestr(member, z.read(member))
                    row["uplink"] = observed
                for cond in ("immediate", "waited"):
                    for eng, key in (("qwen","complete"), ("turbo","complete_suffix")):
                        passed = opened is not None and row["uplink"]["readings"][cond][eng][key]
                        counts[cond+"/"+eng].update(cases=1, original=int(case["readings"][cond][eng][key]), delayed=int(passed))
                rows.append(row)
                print(label, flush=True)
            playback.writestr("manifest.json", json.dumps(dict(records=pairs, gap_samples=GAP, rate=16000,
                instructions="Each pair is one nominal-rate external PCM24 track at quarter scale. Play either track against the same synchronized panel and music files from the three-room playback pack.",
                room_tracks=pack.RELEASE+"c161-bella-weak-speaker-gain-room-audio.zip"), indent=2)+"\n")
        assert len(rows) == 15 and len(pairs) == 6
        sources_used = (Path(__file__), PROTOCOL, weak.OUT, pack.OUT, daily.OUT, weak.early.PLAN)
        with ZipFile(work / TRACE.name) as logs:
            assert len(logs.namelist()) == len(set(logs.namelist()))
            files = {name:fixed.digest(logs.read(name)) for name in logs.namelist()}
        P.save(work/OUT.name, dict(records=rows, counts=counts, pairs=pairs, playback_sha256=P.sha(work/PLAY.name),
               files=files, archive_sha256=P.sha(work/TRACE.name), build=previous["build"], qwen=qid, turbo=base["turbo"],
               decoder=base["decoder"], sources={str(p):P.sha(p) for p in sources_used}))
        for path in (PLAY, TRACE, OUT):
            (work/path.name).replace(path)


if __name__ == "__main__":
    main()
