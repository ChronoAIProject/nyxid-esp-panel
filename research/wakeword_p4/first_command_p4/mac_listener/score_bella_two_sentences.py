"""Replay both native first opens and compare the two intended sentences at the receiver."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_two_sentence_rooms as study
import score_bella_confirmed_sentences as check

P, scoring = check.P, check.scoring
OUT = study.PLAN.with_name(study.STEM + ".json")
TRACE = OUT.with_suffix(".zip")


def main():
    plan = P.read(study.PLAN)
    dual, willow = (P.read(study.WORK / (kind+".json")) for kind in ("dual", "willow"))
    assert dual["plan_sha256"] == willow["plan_sha256"] == P.sha(study.PLAN)
    assert P.sha(study.PACK) == plan["playback_sha256"]
    baseline = P.read(check.prior.trial.baseline.OUT)
    sources = {p.name:P.sha(p) for p in (Path(__file__), Path(check.__file__), Path(check.policy.__file__),
           study.PLAN, check.prior.trial.baseline.OUT, scoring.clock.BUILD / "nyx_mac_listener.bin",
           scoring.prior.CODEC / "nyx_first_command.bin", scoring.sender.BUILD / "board.bin")}
    for name in ("nyx_mac_listener.bin", "nyx_first_command.bin", "board.bin"):
        assert sources[name] == baseline["sources"][name]
    proc, model, identity = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert identity == baseline["qwen"] and decoder.opus_get_version_string().decode() == baseline["decoder"]
    readers = proc, model, scoring.prior.reader(baseline["turbo"]), decoder
    cache = study.WORK / ("sender-" + P.sha(study.PLAN)[:20])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    records, counts, files = [], defaultdict(Counter), {}
    with TemporaryDirectory(dir=study.WORK) as folder:
        work = Path(folder)
        with ZipFile(work / TRACE.name, "w", ZIP_DEFLATED) as evidence:
            for kind, src in (("dual",dual), ("willow",willow)):
                trace = study.WORK/(kind+".zip")
                assert P.sha(trace) == src["archive_sha256"]
                with ZipFile(trace) as archive:
                    for name in archive.namelist():
                        raw = archive.read(name)
                        dest = kind + "/" + name
                        evidence.writestr(dest, raw)
                        files[dest] = scoring.sha256(raw).hexdigest()
            for i, (r, n, w) in enumerate(zip(plan["records"], dual["records"], willow["records"], strict=True)):
                assert n["label"] == r["label"] == w["label"] and w["pcm_sha256"] == r["pcm_sha256"]
                assert all(files["dual/"+k] == h for k,h in n["members"].items())
                assert all(files["willow/"+r["label"]+"/"+k] == h for k,h in w["members"].items())
                role = r["role"]
                times = n["events"]["O"] if role == "nyx" else [x[1] for x in w["events"]] if role == "willow" else []
                opened = next(iter(times), None)
                row = dict(label=r["label"], role=role, command=r["command"], first_sentence=r["first_sentence"],
                           opened=opened, command_at=r["end"], dual=n, willow=w, readings={})
                records.append(row)
                c = counts[role]
                c.update(cases=1, opened=opened is not None, nyx_any=bool(n["events"]["O"]), willow_any=bool(w["events"]),
                         timely=opened is not None and r["start"] <= opened < r["end"])
                if opened is None:
                    continue
                ref_path = work / "silent.wav"
                ref = np.zeros(r["samples"], dtype="<i2")
                sf.write(ref_path, ref, 16000, subtype="PCM_16")
                record = dict(r, reference_path=ref_path, reference_sha256=P.digest(ref))
                uplink, shard = scoring.score_case(record, i, opened, readers, work, cache)
                member_hashes = {}
                with ZipFile(shard) as shard_logs:
                    for name in shard_logs.namelist():
                        if name == "record.json":
                            continue
                        value = shard_logs.read(name)
                        evidence.writestr(name,value)
                        files[name] = member_hashes[name] = scoring.sha256(value).hexdigest()
                    for stage in ("capture","opus","sender"):
                        assert member_hashes[r["label"]+"/"+stage+".log"] == uplink[stage+"_uart_sha256"]
                    row["readings"] = check.new_crop(work, shard_logs, dict(uplink,label=r["label"],uart=member_hashes),
                                                     r["command"], role, readers)
                row.update(first=uplink["first"], stop=uplink["stop"], fifo_sha256=uplink["fifo_sha256"])
                want = check.prior.normalized(r["first_sentence"])
                for cond, readings in row["readings"].items():
                    for reader, reading in readings.items():
                        for mode in ("full", "crop", "confirmed"):
                            words = check.prior.normalized(reading[mode])
                            first = any(words[k:k+len(want)] == want for k in range(len(words)))
                            metric = reading["scores"][mode]
                            metric["first_sentence"] = first
                            for k,v in metric.items():
                                c["_".join((cond,reader,mode,k))] += v
        assert len(records) == 12 and all(x["cases"] == 4 for x in counts.values())
        with ZipFile(work/TRACE.name) as logs:
            assert set(files) == set(logs.namelist()) and len(files) == len(logs.namelist())
        P.save(work/OUT.name, dict(records=records, counts=counts, sources=sources, qwen=identity,
               turbo=baseline["turbo"], decoder=baseline["decoder"], archive_sha256=P.sha(work/TRACE.name),
               files=files, qemu_sha256=dual["qemu_sha256"], guest_sha256=dual["guest_sha256"],
               model_sha256=dual["model_sha256"], scope="Four quiet matched two-sentence commands and four erased controls."))
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
