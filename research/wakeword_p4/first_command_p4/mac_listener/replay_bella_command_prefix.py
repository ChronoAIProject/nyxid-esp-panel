"""Reuse the public af_bella WAV unchanged before each of the 24 paired commands."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import cache_sender_p4_case as cases
import check_connected_marker_gains as gains

ch, tx, study = gains.ch, gains.tx, gains.study
CUE = ch.RESULTS / "hye-nyx-af-bella-marker-rms-v1.wav"
OUT = ch.RESULTS / "4090-c161-bella-command-prefix.json"
CUE_SHA = "3adea27b05e1f541b0c525cc3e5a2f18454ae839951c43bb50bbc771d5c0e30b"


def rear_filter(pcm, taps):
    wet = fftconvolve(pcm.astype(float), taps)
    assert np.max(np.abs(wet[len(pcm):])) < 0.5
    return wet[:len(pcm)]


def recordings():
    tracks = ch.read(gains.OUT)
    controls = {(r["batch"], r["source"]["index"], r["source"]["external_db"]): r
                for r in ch.read(gains.direction.OUT)["records"]
                if r["profile"] == "back" and r["role"] == "no_address"}
    lengths = {(r["source"]["index"], r["source"]["external_db"]): r["samples"]
               for r in ch.read(study.timing.OUT)["records"] if r["track"] == "marked" and r["timing"] == "tx80"}
    panel, sample_rate = sf.read(study.mixed.panel.demo.WAVE, dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    filters, _ = gains.direction.filters()
    assert sample_rate == 16000 and ch.sha(gains.PACK) == tracks["pack_sha256"]
    with ZipFile(gains.PACK) as pack:
        entries = {(r["batch"], r["index"]): r for r in json.loads(pack.read("manifest.json"))["records"]}
        anchor, = (r for r in entries.values() if r.get("wake_only", {}).get("file") == "wake/af_bella.wav")
        cue = study.playback.read_wav(pack, anchor["wake_only"])
        assert ch.sha(CUE) == anchor["wake_only"]["sha256"] == CUE_SHA
        for old in tracks["records"]:
            if old["profile"] != "back":
                continue
            s = old["source"]
            key = old["batch"], s["index"]
            entry = entries[key]
            at = entry["command_at"]
            assert at == s["command_at"] and at >= len(cue)
            dry = study.playback.read_wav(pack, entry["connected"])
            fixed = dry.copy()
            fixed[:at-len(cue)] = 0
            fixed[at-len(cue):at] = cue
            absent = fixed.copy()
            absent[:at] = 0
            assert ch.digest(fixed[at:]) == entry["command_sha256"]
            if key == (anchor["batch"], anchor["index"]):
                assert np.array_equal(fixed, dry)
            length = len(dry)
            if key[0] == "first":
                _, (hx, _, _) = tx.diag.room(s["geometry"])
                length = lengths.pop((key[1], s["external_db"])) - 4800 - len(hx) + 1
            mix = study.mixed.room_scene(s, {"length": (entry, range(length))}, panel, v5, music, music_id)
            original, _ = mix(rear_filter(dry, filters["back"]))
            missing, _ = mix(rear_filter(absent, filters["back"]))
            control = controls.pop((*key, s["external_db"]))
            assert ch.digest(original) == old["result"]["pcm_sha256"]
            assert ch.digest(missing) == control["result"]["pcm_sha256"] and not control["result"]["opened"]
            mic, clips = mix(rear_filter(fixed, filters["back"]))
            source = dict(s, start=at - (anchor["command_at"] - anchor["start"]), cue_file_start=at-len(cue),
                          cue_voice="af_bella", command_pcm_sha256=entry["command_sha256"], dry_pcm_sha256=ch.digest(fixed))
            yield old, source, mic, clips, ch.digest(missing)
    assert not controls and not lengths


def run():
    prior = ch.read(gains.GUEST)
    base = {(r["batch"], r["source"]["index"]): r for r in prior["records"] if r.get("paired_back_grid")}
    assert len(base) == 24 and prior["host_sha256"] == ch.sha(gains.OUT)
    builds = {"bca": gains.direction.mixed.panel.middle.WORK / "build"}
    uplink = tx.panel.grid.gap.uplink
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == prior["firmware"] and uplink.firmware() == prior["codec"]
    small, small_id = ch.load_reader()
    proc, model, qwen_id = study.prefix.balanced.original.asr.load()
    assert small_id == prior["small"] and qwen_id == prior["qwen"]
    readers = small, tx.diag.reader(prior["turbo"]), uplink.decoder_api(), proc, model
    native = tx.panel.grid.room.rooms.c161_binding()
    sources = {p.name: ch.sha(p) for p in (CUE, gains.PACK, gains.OUT, gains.GUEST, gains.direction.OUT,
                                            study.timing.OUT, Path(__file__), Path(study.playback.__file__))}
    provenance = dict(study=OUT.stem, sources=sources, **{k: prior[k] for k in ("firmware", "codec", "qwen", "turbo", "small")})
    rows, captured = [], []
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        work = Path(tmp)
        matcher = tx.single.timed.binding(work)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for old, s, mic, clips, absent_sha in recordings():
                result = study.playback.host_result(matcher, native, mic, s["transmit_event"])
                first = next(iter(result["opened"]), None)
                row = dict(batch=old["batch"], source=s, result=result, clipped_samples=clips,
                           no_address_pcm_sha256=absent_sha, paired_back_grid=old["paired_back_grid"],
                           original_timely=old["timely"], original_opened=old["result"]["opened"],
                           timely=first is not None and s["start"] <= first < s["command_at"])
                if old["paired_back_grid"]:
                    old_p4 = base.pop((old["batch"], s["index"]))
                    assert old_p4["source"] == old["source"] and old_p4["opened"] == row["original_opened"]
                    case = len(rows), s, "bca", "marked", mic, np.zeros_like(mic), result
                    board = cases.capture(work, logs, len(captured), case, readers, builds, provenance)
                    captured.append(dict(board, batch=old["batch"], original_index=old_p4["index"]))
                    print("P4", len(captured), "of 24", flush=True)
                rows.append(row)
        assert len(rows) == 72 and len(captured) == 24 and not base
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        report = dict(policy="Right-align the complete released af_bella file with each original command boundary. "
                      "Reuse 72 rear development mixes and identical no-address controls. Replay the original "
                      "24-case level-balanced rear grid on P4 QEMU. Keep original playback gains, panel clock, "
                      "electrical reference, FIFO history, codec and readers.", model=prior["model"], **provenance,
                      records=rows, captures=captured, archive_sha256=ch.sha(work/archive.name))
        ch.save(work/OUT.name, report)
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
