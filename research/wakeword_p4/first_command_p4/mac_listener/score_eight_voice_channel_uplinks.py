"""Replay complete dry and front command uplinks at the already reported opening clocks."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import eight_voice_willow_channels as channels
import score_eight_voice_willow as prior

ch = prior.ch
OUT = channels.OUT.with_name(channels.OUT.stem + "-uplinks.json")


def capture(work, cache, logs, row, source, mic, readers):
    index = row["index"]
    shard = cache / f"{index:03}.zip"
    if not shard.exists():
        with ZipFile(work / shard.name, "w", ZIP_DEFLATED) as part:
            result = prior.recognize(work, part, index, mic, source, row["decision"]["opened"], readers)
            part.writestr("capture.json", json.dumps(result))
        (work / shard.name).replace(shard)
    with ZipFile(shard) as part:
        assert part.testzip() is None
        for name in part.namelist():
            if name != "capture.json":
                logs.writestr(name, part.read(name))
        return json.loads(part.read("capture.json"))


def main():
    plan, clocks = ch.read(channels.PLAN), ch.read(channels.OUT)
    assert clocks["plan_sha256"] == ch.sha(channels.PLAN)
    baseline = ch.read(prior.OUT)
    codec = prior.uplink.firmware()
    small, small_id = ch.load_reader()
    processor, model, qid = prior.gain.study.prefix.balanced.original.asr.load()
    assert (codec, small_id, qid) == (baseline["codec"], baseline["small"], baseline["qwen"])
    readers = small, prior.gain.tx.diag.reader(baseline["turbo"]), prior.uplink.decoder_api(), processor, model
    sources = {path.name: ch.sha(path) for path in (
        Path(__file__), Path(prior.__file__), Path(prior.uplink.__file__), channels.OUT, channels.PLAN, prior.OUT)}
    cache = channels.WORK / "uplinks" / (sources[channels.OUT.name] + "-" + sources[Path(__file__).name])
    cache.mkdir(parents=True, exist_ok=True)
    archive = OUT.with_suffix(".zip")
    rows, summary = [], defaultdict(Counter)
    with TemporaryDirectory(dir=cache) as tmp:
        work = Path(tmp)
        prior.uplink.WORK = work
        with ZipFile(work / archive.name, "w", ZIP_DEFLATED) as logs:
            for i, (clock, src) in enumerate(zip(clocks["records"], plan["records"], strict=True)):
                role = src["target_id"]
                assert (clock["index"], clock["role"], clock["pcm_sha256"]) == (i, role, src["pcm_sha256"])
                system = "willow" if role == "willow_matched" else "nyx"
                decision = clock[system]
                row = dict(index=i, mode=src["mode"], role=role, batch=src["batch"], track=src["track"],
                           voice=src["voice"], command=src["carrier"], mic_sha256=src["pcm_sha256"],
                           command_at=src["end"], decision=decision)
                stats = summary[src["mode"] + "." + role]
                stats.update(trials=1, opened=int(decision["opened"] is not None), timely=int(decision["timely"]))
                if decision["opened"] is not None:
                    path = Path(src["path"])
                    mic, rate = sf.read(path, dtype="int16")
                    assert rate == 16000 and ch.sha(path) == src["sha256"] and ch.digest(mic) == row["mic_sha256"]
                    result = capture(work, cache, logs, row, src, mic, readers)
                    fifo = result["fifo"]
                    assert fifo["first"] == max(0, decision["opened"] // 160 * 160 - 32000)
                    row["command_offset_in_selected_input"] = src["end"] + 1440 - fifo["first"]
                    assert 0 <= row["command_offset_in_selected_input"] < fifo["emitted"] - fifo["first"]
                    row["capture"] = result
                    for name, field in (("qwen", "complete"), ("turbo", "complete_suffix")):
                        full = int(result[name][field])
                        stats.update({name: full, "timely_" + name: full * decision["timely"],
                                      "late_" + name: full * (not decision["timely"])})
                rows.append(row)
                if (i + 1) % 12 == 0:
                    print("scored", i + 1, flush=True)
            for path in sorted(work.glob("*-attempt*.gz")):
                logs.write(path, "codec-retries/" + path.name)
        with ZipFile(work / archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        assert len(rows) == 144 and sum("capture" in row for row in rows) == 76
        ch.save(work / OUT.name, dict(records=rows, summary=summary, sources=sources, codec=codec, small=small_id,
            turbo=baseline["turbo"], qwen=qid, archive_sha256=ch.sha(work / archive.name),
            policy="Previously measured native Nyx and official Willow opening clocks. Simulate the same two-second "
                   "history, 10 ms FIFO and 90 ms selected-channel delay in both conditions. Encode selected and raw "
                   "inputs on simulated P4. Transcribe the complete decoded selected uplink without command prompts. "
                   "Unopened trials contribute zero to end-to-end counts. Reused synthetic development recordings."))
        (work / archive.name).replace(archive)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
