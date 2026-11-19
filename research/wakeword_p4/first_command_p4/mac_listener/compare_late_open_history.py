"""Re-encode all late dry/front openings with zero history on the same selected channel."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import score_eight_voice_channel_uplinks as full
import score_paired_piper_willow as paired

ch, prior = full.ch, full.prior
OUT = full.OUT.with_name(full.channels.OUT.stem + "-late-history.json")


def main():
    report = ch.read(full.OUT)
    plan = ch.read(full.channels.PLAN)
    sources = {p.name: ch.sha(p) for p in (Path(__file__), Path(full.__file__), Path(prior.__file__),
               Path(paired.__file__), full.OUT, full.channels.PLAN)}
    assert sources[full.channels.PLAN.name] == report["sources"][full.channels.PLAN.name]
    assert ch.sha(full.OUT.with_suffix(".zip")) == report["archive_sha256"]
    small, small_id = ch.load_reader()
    proc, model, qid = prior.gain.study.prefix.balanced.original.asr.load()
    assert (small_id, qid, prior.uplink.firmware()) == (report["small"], report["qwen"], report["codec"])
    readers = small, prior.gain.tx.diag.reader(report["turbo"]), prior.uplink.decoder_api(), proc, model
    cache = full.channels.WORK / "zero-history" / sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cache.mkdir(exist_ok=True, parents=True)
    rows, counts = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=cache) as folder:
        work = Path(folder)
        prior.uplink.WORK = work
        with ZipFile(work / archive.name, "w", ZIP_DEFLATED) as logs:
            for row in report["records"]:
                if "capture" not in row or row["decision"]["timely"]:
                    continue
                index = row["index"]
                src = plan["records"][index]
                mic, sr = sf.read(src["path"], dtype="int16")
                assert sr == 16000 and ch.digest(mic) == row["mic_sha256"] == src["pcm_sha256"]
                before = row["capture"]
                after = full.capture(work, cache, logs, row, src, mic, readers, history_samples=0)
                first = after["fifo"]["first"]
                assert first == row["decision"]["opened"] // 160 * 160 == before["fifo"]["first"] + 32000
                assert after["fifo"]["emitted"] == before["fifo"]["emitted"]
                offset = row["command_at"] + 1440 - first
                g = counts[row["mode"] + "." + row["role"]]
                g.update(trials=1, starts_after_command=int(offset < 0))
                for reader, key in (("qwen", "complete"), ("turbo", "complete_suffix")):
                    old, new = before[reader][key], after[reader][key]
                    g.update({reader + "_history": int(old), reader + "_zero": int(new),
                              reader + "_lost": int(old and not new), reader + "_gained": int(new and not old)})
                rows.append(dict(index=index, mode=row["mode"], role=row["role"], command=row["command"],
                                 mic_sha256=row["mic_sha256"], opened=row["decision"]["opened"],
                                 command_at=row["command_at"], zero_history_command_offset=offset, capture=after))
            for path in sorted(work.glob("*-attempt*.gz")):
                logs.write(path, "codec-retries/" + path.name)
        assert len(rows) == 20
        with ZipFile(work / archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        ch.save(work / OUT.name, dict(records=rows, summary=counts, sources=sources, archive_sha256=ch.sha(work/archive.name),
            policy="Select all late openings in the previous dry/front experiment. Re-encode with the same OPEN, "
                   "microphone, 90 ms channel delay, 10 ms FIFO production clock and final sample. Omit only the "
                   "32,000 samples of FIFO history. A fresh Opus encoder starts at the first retained sample in each arm."))
        (work / archive.name).replace(archive)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
