"""Replay two marked-call cohorts with measured changes in speaker response by azimuth."""
import argparse
from base64 import b64decode
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import cache_sender_p4_case as cases

import check_additional_connected_voices as study

ch, tx, mixed = study.ch, study.tx, study.mixed
OUT = ch.RESULTS / "4090-c161-sender-directivity.json"
GUEST = OUT.with_name(OUT.stem + "-p4.json")
CURVES = ch.RESULTS / "nest-audio-asr-horizontal-curves.json"
BASE_URL = "https://www.spinorama.org/speakers/Google%20Nest%20Audio/ASR/"
GRAPH_SHA = "9f56ca5034da4ae10c12c64d567b1ef8313167ab06bf93ba41e4fe90507d5f38"
ANGLES = ("front", "side", "back")
RATE = 16000


def extract_curves(plot):
    assert ch.sha(plot) == GRAPH_SHA
    traces = {line["name"]: line for line in ch.read(plot)["data"]}
    curves = {}
    for profile, label in zip(ANGLES, ("ON", "90°", "180°"), strict=True):
        line = traces[label]
        assert line["x"]["dtype"] == line["y"]["dtype"] == "f8"
        curves[profile] = {field: np.frombuffer(b64decode(line[axis]["bdata"]), dtype="<f8").tolist()
                           for field, axis in (("hz", "x"), ("db", "y"))}
    CURVES.write_text(json.dumps(dict(
        speaker="Google Nest Audio", measurement="Amir Majidimehr, Audio Science Review",
        review="https://www.audiosciencereview.com/forum/index.php?threads/google-nest-audio-speaker-review.58134/",
        plot=BASE_URL + "asr/SPL%20Horizontal.json", plot_sha256=GRAPH_SHA,
        license="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        change="Three original horizontal SPL traces decoded from the Spinorama plot into numeric arrays.",
        degrees=dict(front=0, side=90, back=180), curves=curves), indent=2) + "\n")


def filters():
    data = ch.read(CURVES)
    assert data["plot_sha256"] == GRAPH_SHA
    ref = data["curves"]["front"]
    shared_db = float(np.mean([v for f, v in zip(ref["hz"], ref["db"], strict=True) if 500 <= f <= 2000]))
    size = 16384
    hz = np.fft.rfftfreq(size, 1/RATE)
    taps, details = {}, {}
    for angle, curve in data["curves"].items():
        frequencies = np.array(curve["hz"])
        assert np.all(np.diff(frequencies) > 0) and frequencies[-1] > RATE/2
        db = np.interp(np.log(np.maximum(hz, frequencies[0])), np.log(frequencies), curve["db"]) - shared_db
        # Reconstruct a causal minimum-phase response from the published magnitudes.
        cepstrum = np.fft.irfft(db * np.log(10)/20)
        cepstrum[1:size//2] *= 2
        cepstrum[size//2+1:] = 0
        impulse = np.fft.irfft(np.exp(np.fft.rfft(cepstrum)))
        taps[angle] = impulse[:2048].copy()
        measured = 20*np.log10(np.abs(np.fft.rfft(taps[angle], size)))
        band = (hz >= 100) & (hz <= 7800)
        error = float(np.max(np.abs(measured[band] - db[band])))
        assert error < 0.3
        marker_band = (hz >= 700) & (hz <= 1580)
        details[angle] = dict(max_error_db_100_7800=error, shared_reference_db=shared_db,
                              marker_band_mean_db=float(np.mean(db[marker_band])),
                              impulse_sha256=sha256(taps[angle].astype("<f8").tobytes()).hexdigest())
    return taps, details


def scenes():
    panel, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    taps, _ = filters()
    for batch, source, pack_path, variant, absent in (
            ("first", study.timing.OUT, study.prefix.balanced.PLAY, "balanced_louder", "marked_no_address"),
            ("second", study.OUT, study.PACK, "marked", "no_address")):
        report = ch.read(source)
        pack_sha = report["sources"][pack_path.name] if batch == "first" else report["pack_sha256"]
        assert ch.sha(pack_path) == pack_sha
        original = {(r["source"]["index"], r["source"]["external_db"], r["track"]): r
                    for r in report["records"] if r.get("timing", "tx80") == "tx80" and r["track"] in ("marked", absent)}
        with ZipFile(pack_path) as pack:
            manifest = json.loads(pack.read("manifest.json"))
            assert rate == RATE and ch.sha(mixed.panel.demo.WAVE) == manifest["panel_sha256"]
            for position, entry in enumerate(manifest["records"]):
                marked = study.playback.read_wav(pack, entry["tracks"][variant])
                removed = marked.copy()
                removed[:entry["command_at"]] = 0
                tracks = {}
                for role, wave in (("marked", marked), ("no_address", removed)):
                    tracks["flat", role] = wave
                    for angle in ANGLES:
                        # The source file ends in silence. Check that dropping its filtered tail is immaterial.
                        wet = fftconvolve(wave.astype(float), taps[angle])
                        assert np.max(np.abs(wet[len(wave):])) < 0.5
                        tracks[angle, role] = wet[:len(wave)]
                for level in mixed.study.LEVELS:
                    old = {role: original.pop((entry["index"], level, key))
                           for role, key in (("marked", "marked"), ("no_address", absent))}
                    s = old["marked"]["source"]
                    assert s["transmit_event"] == s["command_at"] - 1280
                    # The first pack used the longest unmodified name variant to set its music interval.
                    if batch == "first":
                        (_, _, _), (hx, _, _) = tx.diag.room(s["geometry"])
                        length = old["marked"]["samples"] - 4800 - len(hx) + 1
                    else:
                        length = len(marked)
                    assert length >= len(marked)
                    mix = mixed.room_scene(s, {"length": (entry, range(length))}, panel, v5, music, music_id)
                    for (angle, role), wave in tracks.items():
                        mic, clipped = mix(wave)
                        ref = old[role].get("baseline", old[role].get("result"))
                        if angle == "flat":
                            assert ch.digest(mic) == ref["pcm_sha256"]
                        yield batch, position, s, angle, role, mic, clipped, ref
        assert not original


def host():
    counts, rows = defaultdict(Counter), []
    native = tx.panel.grid.room.rooms.c161_binding()
    ref = ch.read(study.timing.GUEST)
    assert ref["model"] == ch.read(study.GUEST)["model"] == ch.read(study.OUT)["c161"]
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        for batch, pos, s, angle, role, mic, clipped, old in scenes():
            result = study.playback.host_result(matcher, native, mic, s["transmit_event"])
            if angle == "flat":
                assert result == old
            total = counts[f"{batch}.{angle}.{role}"]
            timely = mixed.study.tally(total, s, result)
            selected_level = mixed.study.LEVELS[(pos//3 + pos%3) % 3]
            board = angle != "flat" and ((role == "marked" and s["external_db"] == selected_level)
                    or (role == "no_address" and angle == "back" and s["external_db"] == -6))
            rows.append(dict(batch=batch, source=s, profile=angle, role=role, result=result, timely=timely,
                             selected_for_p4=board, clipped_samples=clipped))
        assert len(rows) == 576 and sum(r["selected_for_p4"] for r in rows) == 96
        paths = (CURVES, study.timing.OUT, study.OUT, study.prefix.balanced.PLAY, study.PACK,
                 mixed.panel.demo.WAVE, Path(mixed.__file__), Path(study.playback.__file__), Path(__file__))
        ch.save(work/OUT.name, dict(records=rows, summary=counts, filters=filters()[1],
                sources={p.name: ch.sha(p) for p in paths}, model=ref["model"], threshold=0.91796875,
                smoothing_steps=5, policy="Apply a 2048-tap minimum-phase fit to ASR horizontal SPL at 0, 90, "
                "180 degrees before the existing room response. All directions share one front-reference level. "
                "Fixed sender command, background, panel TX, preamp. Zero electrical reference. "
                "Same consumed paired development recordings. Select one balanced level per source for all "
                "three directions and all rear weakest-level no-address controls for P4 before ASR."))
        (work/OUT.name).replace(OUT)


def guest():
    host_report = ch.read(OUT)
    first, second = ch.read(study.timing.GUEST), ch.read(study.GUEST)
    assert all(first[k] == second[k] for k in ("model", "firmware", "codec", "small", "turbo", "qwen"))
    uplink = tx.panel.grid.gap.uplink
    builds = dict(bca=mixed.panel.middle.WORK/"build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == first["firmware"] and uplink.firmware() == first["codec"]
    small, sid = ch.load_reader()
    proc, model, qid = study.prefix.balanced.original.asr.load()
    assert sid == first["small"] and qid == first["qwen"]
    readers = small, tx.diag.reader(first["turbo"]), uplink.decoder_api(), proc, model
    archive = GUEST.with_suffix(".zip")
    provenance = dict(study=GUEST.stem, host=ch.sha(OUT), firmware=first["firmware"], codec=first["codec"],
                      qwen=qid, small=sid, turbo=first["turbo"], capture=ch.sha(Path(study.playback.__file__)))
    rows = []
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for host_index, (scene, known) in enumerate(zip(scenes(), host_report["records"], strict=True)):
                batch, _, s, angle, role, mic, _, _ = scene
                assert (batch, s, angle, role, ch.digest(mic)) == (
                    known["batch"], known["source"], known["profile"], known["role"], known["result"]["pcm_sha256"])
                if not known["selected_for_p4"]:
                    continue
                i = len(rows)
                case = host_index, s, "bca", role, mic, np.zeros_like(mic), known["result"]
                row = cases.capture(work, logs, i, case, readers, builds, provenance)
                rows.append(dict(row, batch=batch, profile=angle, timely=known["timely"]))
                if len(rows) % 12 == 0:
                    print("P4", len(rows), flush=True)
        assert len(rows) == 96
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        versions = {k: first[k] for k in ("model", "firmware", "codec", "small", "turbo", "qwen")}
        ch.save(work/GUEST.name, dict(records=rows, **versions, host_sha256=ch.sha(OUT),
                flat_reports={p.name: ch.sha(p) for p in (study.timing.GUEST, study.GUEST)},
                archive_sha256=ch.sha(work/archive.name)))
        (work/archive.name).replace(archive)
        (work/GUEST.name).replace(GUEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "host", "p4"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.mode == "extract":
        if not args.plot:
            parser.error("extract requires --plot with the original horizontal SPL JSON")
        extract_curves(args.plot)
    elif args.mode == "host":
        host()
    else:
        guest()
