"""Peak-normalize only the public af_bella call and replay its original command suffixes."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_weak_playback as weak
import package_bella_command_playback as package

bella, ch, tx, study = weak.bella, weak.ch, weak.tx, weak.study
OUT = ch.RESULTS / "4090-c161-bella-peak-playback.json"
GUEST = ch.RESULTS / "4090-c161-bella-peak-playback-p4.json"
CUE = ch.RESULTS / "hye-nyx-af-bella-peak-minus1.wav"
PACK = ch.RESULTS / "c161-bella-peak-command-playback.zip"
WORK = Path(".local/wakeword/c161-bella-peak-playback")


def wav_bytes(pcm):
    buf = BytesIO()
    sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def host():
    baseline = ch.read(bella.OUT)
    assert ch.sha(weak.PACK) == "67b54cc6493d1925914d0628e1785ca693e1b41dc5ee61e18c49f48858fe2e94"
    cue, rate = sf.read(bella.CUE, dtype="int16")
    assert rate == 16000 and ch.sha(bella.CUE) == bella.CUE_SHA
    gain = (32768 * 10**(-1/20)) / np.abs(cue.astype(float)).max()
    gained = np.rint(cue.astype(float)*gain).astype("<i2")
    native = tx.panel.grid.room.rooms.c161_binding()
    panel, rate = sf.read(study.mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    filters, detail = bella.gains.direction.filters()
    lengths = {(r["source"]["index"],r["source"]["external_db"]): r["samples"]
               for r in ch.read(study.timing.OUT)["records"] if r["track"]=="marked" and r["timing"]=="tx80"}
    cases = {(r["batch"],r["source"]["index"],r["source"]["external_db"]):r for r in baseline["records"]}
    WORK.mkdir(exist_ok=True)
    rows, tracks = [], []
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(weak.PACK) as previous:
        temp = Path(folder)
        matcher = tx.single.timed.binding(temp)
        (temp/CUE.name).write_bytes(wav_bytes(gained))
        with ZipFile(temp/PACK.name, "w") as archive:
            for entry in json.loads(previous.read("manifest.json"))["records"]:
                path = Path(entry["file"])
                dry = study.playback.read_wav(previous,entry)
                at, start = entry["command_at"], entry["cue_file_start"]
                assert at-start == len(cue) and np.array_equal(dry[start:at],cue)
                louder = dry.copy()
                louder[start:at] = gained
                assert ch.digest(louder[at:]) == entry["command_pcm_sha256"]
                data = wav_bytes(louder)
                package.put(archive,entry["file"],data)
                tracks.append(dict(file=entry["file"], sha256=package.digest(data), pcm_sha256=ch.digest(louder),
                                   original_sha256=entry["sha256"], cue_file_start=start, command_at=at,
                                   command=entry["command"], command_voice=entry["command_voice"],
                                   command_pcm_sha256=entry["command_pcm_sha256"]))
                removed = dry.copy()
                removed[:at] = 0
                signals = [bella.rear_filter(x, filters["back"]) for x in (dry, removed, louder)]
                for db in (-6, 0, 6):
                    parent = cases.pop((path.parent.name, int(path.stem), db))
                    s = parent["source"]
                    assert s["dry_pcm_sha256"] == entry["pcm_sha256"]
                    length = len(dry)
                    if parent["batch"] == "first":
                        _, (hx, _, _) = tx.diag.room(s["geometry"])
                        length = lengths.pop((s["index"],db)) - 4800 - len(hx) + 1
                    mix = study.mixed.room_scene(s, {"length":(entry, range(length))}, panel, v5, music, music_id)
                    raw, old_clips = mix(signals[0])
                    assert ch.digest(raw) == parent["result"]["pcm_sha256"] and old_clips == parent["clipped_samples"]
                    silent, _ = mix(signals[1])
                    control = study.playback.host_result(matcher, native, silent, s["transmit_event"])
                    assert control["pcm_sha256"] == parent["no_address_pcm_sha256"]
                    mic, clips = mix(signals[2])
                    result = study.playback.host_result(matcher, native, mic, s["transmit_event"])
                    first = next(iter(result["opened"]), None)
                    timely = first is not None and s["start"] <= first < at
                    row = dict(index=len(rows), file=entry["file"], source=dict(s,dry_pcm_sha256=ch.digest(louder)),
                               result=result, timely=timely, old_timely=parent["timely"], old_open=parent["result"]["opened"],
                               clips=clips, original_clips=old_clips, no_address=control, paired_grid=parent["paired_back_grid"])
                    # Replay all original P4 pairs, and every original weak missed case.
                    if row["paired_grid"] or (db == -6 and not row["old_timely"]):
                        sf.write(WORK/f'{row["index"]:03}.wav', mic, 16000, subtype="PCM_16")
                    rows.append(row)
            manifest = dict(format="16 kHz mono PCM16", clocks="Zero-based sample offsets within each file",
                            cue=CUE.name, cue_sha256=ch.sha(temp/CUE.name), original_pack_sha256=ch.sha(weak.PACK),
                            cue_gain_db=float(20*np.log10(gain)), records=tracks)
            package.put(archive,"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
        assert len(rows)==72 and len(tracks)==24 and not cases and not lengths
        with ZipFile(temp/PACK.name) as archive:
            assert archive.testzip() is None
        ch.save(temp/OUT.name, dict(records=rows, input_cue_sha256=bella.CUE_SHA, gain_db=manifest["cue_gain_db"],
                sample_peak=int(np.abs(gained).max()), cue_sha256=manifest["cue_sha256"], pack_sha256=ch.sha(temp/PACK.name),
                model=baseline["model"], previous_sha256=ch.sha(bella.OUT), filters=detail,
                policy="Amplify only the released call to -1 dBFS sample peak. Identical command, mic gain, "
                       "background, panel, simulated receiver and clocks across each comparison."))
        for dest in (CUE, PACK, OUT):
            (temp/dest.name).replace(dest)


def p4():
    prior = ch.read(bella.OUT)
    plan = ch.read(OUT)
    builds = dict(bca=bella.gains.direction.mixed.panel.middle.WORK / "build")
    uplink = tx.panel.grid.gap.uplink
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == prior["firmware"] and uplink.firmware() == prior["codec"]
    small, sid = ch.load_reader()
    proc, model, qid = study.prefix.balanced.original.asr.load()
    assert sid == prior["small"] and qid == prior["qwen"]
    readers = small, tx.diag.reader(prior["turbo"]), uplink.decoder_api(), proc, model
    previous = {(r["batch"],r["source"]["index"]):r for r in prior["captures"]}
    provenance = dict(study=GUEST.stem, plan_sha256=ch.sha(OUT), **{k:prior[k] for k in ("firmware","codec","qwen","turbo","small")})
    rows = []
    archive = GUEST.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        with ZipFile(temp/archive.name, "w", ZIP_DEFLATED) as logs:
            for r in plan["records"]:
                s = r["source"]
                if not (r["paired_grid"] or s["external_db"] == -6 and not r["old_timely"]):
                    continue
                mic, rate = sf.read(WORK/f'{r["index"]:03}.wav', dtype="int16")
                assert rate==16000 and ch.digest(mic)==r["result"]["pcm_sha256"]
                case = r["index"], s, "bca", "marked", mic, np.zeros_like(mic), r["result"]
                row = bella.cases.capture(temp, logs, len(rows), case, readers, builds, provenance)
                row.update(file=r["file"], paired_grid=r["paired_grid"], original_open=r["old_open"])
                if r["paired_grid"]:
                    old = previous.pop((Path(r["file"]).parent.name,s["index"]))
                    assert s["external_db"]==old["source"]["external_db"] and r["old_open"]==old["opened"]
                    row.update(original_qwen=old["qwen"], original_turbo=old["uplink"]["turbo"])
                rows.append(row)
                print("P4",len(rows),r["file"],row["opened"],flush=True)
        assert not previous and len(rows)==26
        with ZipFile(temp/archive.name) as logs:
            assert logs.testzip() is None
        ch.save(temp/GUEST.name, dict(records=rows, archive_sha256=ch.sha(temp/archive.name), **provenance))
        (temp/archive.name).replace(archive)
        (temp/GUEST.name).replace(GUEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("host","p4"))
    args = parser.parse_args()
    (host if args.step == "host" else p4)()
