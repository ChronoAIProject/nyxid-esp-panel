"""Play the four-voice synthesis through sample-rate conversion, AAC and SBC before the mic mix."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_connected_mic_headroom as headroom
import check_marked_connected_cues as cues
import handoff_playback_codecs as playback

ch, tx, mixed, balanced = headroom.ch, headroom.tx, headroom.mixed, headroom.balanced
OUT = ch.RESULTS / "4090-c161-connected-playback-codecs.json"


def read_wav(pack, entry):
    data = pack.read(entry["file"])
    assert sha256(data).hexdigest() == entry["sha256"]
    pcm, rate = sf.read(BytesIO(data), dtype="int16")
    assert rate == 16000 and pcm.ndim == 1 and ch.digest(pcm) == entry["pcm_sha256"]
    return pcm


def versions(pcm, calibration):
    for profile in playback.PROFILES:
        decoded, info = playback.encode_decode(np.pad(pcm, (0, 2048)), profile)
        lag = calibration[profile]["lag"]
        assert len(decoded) >= len(pcm) + lag
        aligned = decoded[lag:lag+len(pcm)]
        yield profile, 0, aligned, info
        if lag:
            yield profile + "_delayed", lag, decoded[:len(pcm)+lag], info


def host_result(matcher, native, mic, clock):
    raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, mic)
    echoes = mixed.panel.middle.single.hits(mic, clock) if clock else []
    return mixed.panel.middle.pair.admit(raw, clock, echoes, np.zeros_like(mic))


def standalone(matcher, native, calibration):
    previous = ch.read(cues.OUT)
    assert ch.sha(cues.PACK) == previous["pack_sha256"]
    records = []
    with ZipFile(cues.PACK) as pack:
        for old in previous["records"]:
            meta = old["source"]
            pcm = read_wav(pack, old)
            for profile, lag, played, info in versions(pcm, calibration):
                mic = np.pad(played, (32000, 32000))
                host = host_result(matcher, native, mic, 0)
                opens = host["opened"]
                if profile == "raw":
                    assert host["pcm_sha256"] == old["mic_sha256"] and opens == old["opened"]
                timely = bool(opens and meta["start"]+lag <= opens[0] < meta["command_at"]+lag)
                records.append(dict(file=old["file"], profile=profile, lag_samples=lag, timely=timely,
                                    command_at=meta["command_at"]+lag, result=host, encoder=info))
    return records


def scenes(prior, calibration):
    rooms = mixed.study.sources()
    v5, _ = tx.study.panel_wave()
    quiet, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000 and ch.sha(mixed.panel.demo.WAVE) == prior["sources"][mixed.panel.demo.WAVE.name]
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    assert ch.sha(balanced.PLAY) == prior["sources"][balanced.PLAY.name]
    original = {(r["source"]["index"],r["source"]["external_db"],r["role"]):r for r in prior["records"]}
    with ZipFile(balanced.PLAY) as pack:
        for entry in json.loads(pack.read("manifest.json"))["records"]:
            wav = read_wav(pack, entry["tracks"]["balanced_louder"])
            no_address = wav.copy()
            no_address[:entry["command_at"]] = 0
            for role, dry in (("both", wav), ("no_address", no_address)):
                tracks = list(versions(dry, calibration))
                for db in mixed.study.LEVELS:
                    old = original.pop((entry["index"], db, role))
                    s = old["source"]
                    assert entry["command_at"] == s["command_at"]
                    render = mixed.room_mix(s, rooms[s["voice"], s["command"]], quiet, v5, bed, bed_sha)
                    for profile, lag, played, info in tracks:
                        meta = dict(s, start=s["start"]+lag, command_at=s["command_at"]+lag)
                        mic, clipped = mixed.digitize(render(played), s["preamp_gain"])
                        if profile == "raw":
                            assert ch.digest(mic) == old["mic_sha256"] and clipped == old["clipped_samples"]
                        yield profile, role, meta, mic, clipped, old, info, ch.digest(played)
    assert not original


def guest_case(work, archive, index, case, readers, builds):
    small, turbo, decoder, proc, model = readers
    _, meta, _, _, mic, _, _ = case
    row = tx.panel.capture(work, archive, index, case, small, turbo, decoder, builds, tx_pair=meta["transmit_event"])
    if row["opened"]:
        cap = row["uplink"]
        log = archive.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
        uplink = tx.panel.grid.gap.uplink
        packets = uplink.packets(log, index, cap["samples"], 0)
        audio = uplink.decode(decoder, packets)
        assert ch.digest(audio) == cap["pcm_sha256"] and len(packets) == cap["packets"]
        assert ch.digest(balanced.fifo_input(mic, row["finish"])) == cap["fifo_sha256"]
        row["qwen"] = balanced.evaluate(proc, model, audio, meta["command"])
    return row


def main():
    prior = ch.read(headroom.OUT)
    calibration = playback.latencies()
    history = ch.read(playback.OUT.with_suffix(".json"))
    assert calibration == history["alignment"]
    versions_used = dict(pyav=playback.av.__version__, libraries=playback.av.library_versions)
    builds = dict(bca=mixed.panel.middle.WORK/"build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == prior["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == prior["codec"]
    small, identity = ch.load_reader()
    turbo = tx.diag.reader(prior["turbo"])
    proc, model, qid = balanced.original.asr.load()
    assert identity == prior["small"] and qid == prior["qwen"]
    readers = small, turbo, tx.panel.grid.gap.uplink.decoder_api(), proc, model
    sources = {p.name:ch.sha(p) for p in (headroom.OUT, balanced.PLAY, cues.OUT, cues.PACK,
               Path(mixed.__file__), Path(playback.__file__), Path(tx.panel.__file__), Path(__file__))}
    key = sha256(json.dumps([sources,versions_used], sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(".local/wakeword/c161-connected-playback-codecs")/key
    cache.mkdir(exist_ok=True, parents=True)
    rows, guest, summary = [], [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=cache) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        isolated = standalone(matcher, native, calibration)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for profile, role, s, mic, clipped, old, info, played_hash in scenes(prior, calibration):
                expected = host_result(matcher, native, mic, s["transmit_event"])
                if profile == "raw":
                    assert expected["opened"] == old["opened"] and expected["allowed_fixed"] == old["allowed_fixed"]
                label = f'{profile}.{role}.{s["external_db"]}'
                tally = summary[label]
                timely = mixed.study.tally(tally, s, expected)
                tally.update(clipped=clipped)
                host_index = len(rows)
                rows.append(dict(source=s, profile=profile, role=role, played_pcm_sha256=played_hash, encoder=info,
                                 original_index=old["index"], result=expected, timely=timely, clipped_samples=clipped))
                # Physical playback cannot be assumed to compensate the codec's bulk delay.
                if not profile.endswith("_delayed") or (role != "both" and s["external_db"] != -6):
                    continue
                index = len(guest)
                shard = cache/f"{index:03}.zip"
                if not shard.exists():
                    case = host_index, s, "bca", role, mic, np.zeros_like(mic), expected
                    part = work/shard.name
                    with ZipFile(part, "w", ZIP_DEFLATED) as case_log:
                        row = guest_case(work, case_log, index, case, readers, builds)
                        case_log.writestr("case.json", json.dumps(row))
                    part.replace(shard)
                with ZipFile(shard) as case_log:
                    row = json.loads(case_log.read("case.json"))
                    assert (row["host_index"], row["source"], row["opened"], row["mic_sha256"]) == (
                        host_index, s, expected["opened"], expected["pcm_sha256"])
                    for member in case_log.namelist():
                        if member != "case.json":
                            logs.writestr(member, case_log.read(member))
                guest.append(dict(row, profile=profile, timely=timely, previous_index=old["index"]))
                if role == "both":
                    before = old["qwen"]["complete"]
                    after = bool(row.get("qwen", {}).get("complete"))
                    tally.update(qwen=int(after), previous_qwen=int(before), gained=int(after and not before),
                                 lost=int(before and not after), turbo=int(row["turbo"]), previous_turbo=int(old["turbo"]))
                if len(guest) % 12 == 0:
                    print("P4 captures", len(guest), flush=True)
        assert len(isolated) == 84 and len(rows) == 504 and len(guest) == 96
        with ZipFile(work/archive.name) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        report = dict(records=rows, guest=guest, standalone=isolated, summary=summary, sources=sources,
                      calibration=calibration, profiles=playback.PROFILES, codec_versions=versions_used,
                      firmware=prior["firmware"], model=prior["model"], codec=prior["codec"], small=identity,
                      turbo=prior["turbo"], qwen=qid, archive_sha256=ch.sha(work/archive.name),
                      policy="Encode only the external speaker. Check native, PCM 44.1/48 kHz, AAC and SBC. "
                             "Compressed variants either remove seeded-probe bulk lag or retain it and shift only "
                             "the external timing labels. Fixed panel TX, music and fixed gain from the prior study. "
                             "Mute the address before encoding for controls. All delayed compressed positives and "
                             "their weakest-level controls get firmware replays. Complete uplink ASR, closed counts "
                             "as a miss. Local file codec simulation; no physical Bluetooth buffering or RF losses.")
        ch.save(work/OUT.name, report)
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
