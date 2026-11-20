"""Read full peer sends at paired playback levels and check the fixed sound's command boundary."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import bella_third_triplet_player_level as level
import score_third_triplet_command_start as anchor
from score_third_triplet_commands import P, SENDER, chosen, scoring, triple


def read_words(readers, segment, target):
    proc, model, whisper, _ = readers
    return dict(qwen=scoring.prior.evaluate(proc, model, segment, target)["text"],
                turbo=scoring.prior.recognition(whisper, segment, target)["text"])


def cropped_readings(record, index, uplink, shard, readers, work, cue, *,
                     locate=anchor.alignment_tool.locate_end, previous_readings=None):
    mic, rate = sf.read(record["path"], dtype="int16")
    assert rate == 16000 and P.digest(mic) == record["pcm_sha256"]
    end, similarity = locate(mic, uplink["opened"], cue)
    name = record["label"]
    with ZipFile(shard) as saved:
        codec = saved.read(name+"/opus.log")
        assert sha256(codec).hexdigest() == uplink["opus_uart_sha256"]
        log = work/"peer.log"
        log.write_bytes(saved.read(name+"/sender.log"))
    assert P.sha(log) == uplink["sender_uart_sha256"]
    packets = scoring.prior.packets(codec.decode().replace("\r\n","\n"), index, uplink["samples"], 0)
    sends = scoring.sender.compare(log, index, uplink["opened"], uplink["first"], uplink["samples"], packets)
    readings = {}
    for schedule, (frames, meta) in zip(("immediate", "waited"), sends, strict=True):
        pcm = scoring.prior.decode(readers[-1], frames)
        assert P.digest(pcm) == uplink["readings"][schedule]["qwen"]["pcm_sha256"]
        cut = max(0, end - 1600 - uplink["first"] + 320 * meta["sent_silence"])
        assert cut < len(pcm)
        digest = P.digest(pcm[cut:])
        previous = (previous_readings or {}).get(schedule)
        if previous and previous["pcm_sha256"] == digest:
            text = previous["text"]
        else:
            text = read_words(readers, pcm[cut:], record["command"])
        readings[schedule] = dict(text=text, cut_sample=cut, pcm_sha256=digest)
    return dict(estimated_end=end, error_samples=end-record["end"], similarity=similarity, readings=readings)


def count_readings(stats, row):
    wanted = anchor.words(row["command"])
    for schedule, pair in row["uplink"]["readings"].items():
        for reader in ("qwen", "turbo"):
            def tally(variant, text):
                actual = anchor.words(text)
                prefix = f"{schedule}_{reader}_{variant}"
                stats[prefix+"_whole"] += (actual == wanted)
                stats[prefix+"_suffix"] += (actual[-len(wanted):] == wanted)
            tally("full", pair[reader]["text"])
            if "crop" in row:
                tally("cropped", row["crop"]["readings"][schedule]["text"][reader])


def main(dataset):
    wake = level.angles.OUT if dataset == "original" else level.OUT
    plan = level.angles.PLAN if dataset == "original" else level.PLAN
    report = P.read(wake)
    assert P.sha(plan) == report["plan_sha256"] and P.sha(P.QEMU) == report["build"]["qemu_sha256"]
    evidence = P.read(anchor.trial.OUT)
    scoring.sender.BUILD = SENDER
    images = (scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin", SENDER/"board.bin")
    assert all(P.sha(p) == evidence["sources"][str(p)] for p in images)
    cue_path = triple.wave(chosen.NAME)
    cue, rate = sf.read(cue_path, dtype="int16")
    assert rate == 16000 and P.sha(cue_path) == P.read(triple.OUT)["voices"][chosen.NAME]
    proc, model, qid = scoring.prior.load()
    assert qid == evidence["qwen"] and "4090" in qid["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == evidence["decoder"]
    readers = proc, model, scoring.prior.reader(evidence["turbo"]), decoder
    out = level.OUT.with_stem(level.OUT.stem+"-"+dataset+"-commands")
    archive = level.WORK/out.with_suffix(".zip").name
    cache = level.WORK/("sends-"+P.sha(wake)[:16]+"-"+P.sha(images[-1])[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, counts, files = [], defaultdict(Counter), {}
    with TemporaryDirectory(dir=level.WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for i, r in enumerate(report["records"]):
                role = r["role"]
                if role == "absent":
                    continue
                engine = "willow" if role == "willow" else "nyx"
                opened = r[engine+"_open"]
                row = {k:r[k] for k in ("label","role","angle","direction","scene","command","end")}
                mode = "original" if dataset == "original" else r["mode"]
                row.update(mode=mode, opened=opened, timely=r[engine+"_timely"])
                stats = counts[mode+"/"+role]
                stats.update(cases=1, opened=int(opened is not None), timely=int(row["timely"]))
                if opened is not None:
                    u, shard = scoring.score_case(dict(r, reference_sha256=r["ref_sha256"]), i, opened, readers, work, cache)
                    row["uplink"] = u
                    if role == "three-triplets":
                        row["crop"] = cropped_readings(r, i, u, shard, readers, work, cue)
                    count_readings(stats, row)
                    with ZipFile(shard) as saved:
                        for member in saved.namelist():
                            if member.endswith(".log"):
                                assert member not in files
                                data = saved.read(member)
                                files[member] = sha256(data).hexdigest()
                                logs.writestr(member, data)
                rows.append(row)
                print(len(rows), r["label"], opened, flush=True)
        assert all(t["cases"] == 12 for t in counts.values())
        assert len(rows) == (36 if dataset=="original" else 72)
        assert sum(not f.startswith("retry/") for f in files) == 3*sum(r["opened"] is not None for r in rows)
        deps = (Path(__file__).resolve().relative_to(Path.cwd()), anchor.SCRIPT,
                Path(anchor.alignment_tool.__file__).resolve().relative_to(Path.cwd()),
                level.PROTOCOL, wake, plan, anchor.trial.OUT, cue_path, *images)
        P.save(work/out.name, dict(records=rows, counts=counts, files=files, archive=archive.name,
                                   archive_sha256=P.sha(work/archive.name), sources={str(p):P.sha(p) for p in deps},
                                   qwen=qid, turbo=evidence["turbo"], decoder=evidence["decoder"]))
        (work/archive.name).replace(archive)
        (work/out.name).replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("original","reduced"))
    main(parser.parse_args().dataset)
