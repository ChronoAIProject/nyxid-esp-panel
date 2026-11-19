"""Move only the panel sound and its playback event relative to a connected command."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_connected_playback_codecs as playback
import score_connected_overlap as overlap

ch, mixed, tx = playback.ch, playback.mixed, playback.tx
OUT = ch.RESULTS / "4090-c161-connected-panel-timing.json"
GUEST = OUT.with_name(OUT.stem + "-p4.json")
TIMINGS = ("original", "tx160", "tx80", "tx0")


def scenes():
    plain_report = ch.read(overlap.OUT)
    marked_report = ch.read(playback.OUT)
    assert ch.sha(playback.balanced.PLAY) == marked_report["sources"][playback.balanced.PLAY.name]
    source = {(r["source"]["index"], r["source"]["external_db"]): r["source"]
              for r in plain_report["records"] if r["source"]["target"] == "Hey Nyx" and r["lane"] == "both"}
    marked = {(r["source"]["index"], r["source"]["external_db"]): r["source"]
              for r in marked_report["records"] if r["profile"] == "raw" and r["role"] == "both"}
    groups = mixed.study.sources()
    panel, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000
    v5, _ = tx.study.panel_wave()
    music, _, music_hash = tx.study.grid.room.rooms.read_noise("music")
    tx_offset = tx.event_time()
    with ZipFile(playback.balanced.PLAY) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert ch.sha(mixed.panel.demo.WAVE) == manifest["panel_sha256"]
        for entry in manifest["records"]:
            coded = playback.read_wav(pack, entry["tracks"]["balanced_louder"])
            for db in mixed.study.LEVELS:
                key = entry["index"], db
                s, fixed = source.pop(key), marked.pop(key)
                assert {k:v for k,v in s.items() if k != "preamp_gain"} == {
                    k:v for k,v in fixed.items() if k != "preamp_gain"}
                group = groups[s["voice"], s["command"]]
                src, plain = group["Hey Nyx"]
                end = s["command_at"]
                assert src["index"] == entry["index"] and src["end"] == end == entry["command_at"]
                (hp, _, _), (hx, _, _) = tx.diag.room(s["geometry"])
                local = fftconvolve(panel.astype(float), hp)
                # Render only the original music on the original time axis so neither the
                # normalization interval nor the tail changes as the panel starts earlier.
                background = mixed.room_mix(s, group, np.zeros_like(panel), v5, music, music_hash)(np.zeros_like(plain))
                waves = dict(plain=plain, marked=coded)
                for name, wave in list(waves.items()):
                    removed = wave.copy()
                    removed[:end] = 0
                    waves[name + "_no_address"] = removed
                waves["panel_music_only"] = np.zeros_like(plain)
                wet = {}
                for name, wave in waves.items():
                    signal = fftconvolve(wave.astype(float), hx) * 10**(db/20)
                    wet[name] = np.pad(signal, (0, len(background)-len(signal)))
                assert s["panel_at"] + tx_offset == s["transmit_event"]
                events = (s["transmit_event"], end-2560, end-1280, end)
                external, clipped = mixed.digitize(background + wet["plain"], s["preamp_gain"])
                yield "external", "plain", dict(s, view="external", transmit_event=0), external, clipped
                for timing, event in zip(TIMINGS, events, strict=True):
                    at = event - tx_offset
                    base = background.copy()
                    assert 0 <= at and at+len(local) <= len(base)
                    base[at:at+len(local)] += local
                    for name, external_sound in wet.items():
                        meta = fixed if name.startswith("marked") else s
                        meta = dict(meta, panel_at=at, transmit_event=event)
                        mic, clipped = mixed.digitize(base + external_sound, meta["preamp_gain"])
                        yield timing, name, meta, mic, clipped
    assert not source and not marked


def check_original(row, clock, codes, old, archive):
    assert (row["pcm_sha256"], row["baseline"]["c161"], row["baseline"]["allowed_fixed"]) == (
        old["pcm_sha256"], old["c161"], old["allowed_fixed"])
    np.testing.assert_array_equal(clock, archive[old["key"]+"_clock"])
    np.testing.assert_array_equal(codes, archive[old["key"]+"_overlap"])
    assert row["checks"] == [c for c in old["checks"] if c["end"] == c["t"]]
    assert row["baseline"]["opened"] == old["opened"]["baseline"]
    for arm in ("overlap", "confirm_0"):
        assert row[arm]["proposals"] == old["proposals"][arm]
        assert row[arm]["opened"] == old["opened"][arm]


def host():
    history = ch.read(overlap.OUT)
    front, ids = overlap.load_models()
    five, five_ids = overlap.verifier.models()
    assert ids == history["models"] and five_ids["trained"] == history["five_sha256"]
    archive = overlap.OUT.with_suffix(".npz")
    assert ch.sha(archive) == history["archive_sha256"]
    lanes = dict(plain="both", plain_no_address="no_address", panel_music_only="panel_music_only")
    old_plain = {(r["source"]["index"], r["source"]["external_db"], r["lane"]):r for r in history["records"]
                 if r["lane"] in (*lanes.values(), "external") and r["source"]["target"] == "Hey Nyx"}
    old_marked = {(r["source"]["index"], r["source"]["external_db"], r["role"]):r
                  for r in ch.read(playback.OUT)["records"] if r["profile"] == "raw"}
    native = overlap.fit.prior.letters.native_frontend()
    c161 = tx.panel.grid.room.rooms.c161_binding()
    frozen = Path(".local/wakeword/causal-source-balanced-int8-v1/int8.tflite")
    assert frozen.stat().st_size == 60952 and ch.sha(frozen) == history["baseline"]
    rows, counts = [], defaultdict(Counter)
    with TemporaryDirectory(dir=OUT.parent) as folder, np.load(archive, allow_pickle=False) as saved:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        for timing, name, s, mic, clipped in scenes():
            event, end = s["transmit_event"], s["command_at"]
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher, c161, mic)
            echo = mixed.panel.middle.single.hits(mic, event)
            zero = np.zeros_like(mic)
            result = mixed.panel.middle.pair.admit(raw, event, echo, zero)
            row = dict(source=s, timing=timing, track=name, samples=len(mic), clipped=clipped,
                       pcm_sha256=raw["pcm_sha256"], prefix_sha256=ch.digest(mic[:end]), echo=echo,
                       baseline=result)
            arms = ["baseline"]
            key = s["index"], s["external_db"]
            if name.startswith("marked"):
                if timing == "original":
                    role = "both" if name == "marked" else "no_address"
                    old = old_marked.pop((*key, role))
                    assert result == old["result"] and clipped == old["clipped_samples"]
            else:
                clock, scores, proposals, checks = overlap.traces({"overlap":front["overlap"]}, native, mic,
                                                                   five["trained"], history["cutoff"])
                row["checks"] = [c for c in checks if c["end"] == c["t"]]
                row["codes_sha256"] = ch.digest(scores["overlap"])
                for arm in ("overlap", "confirm_0"):
                    hits = sorted(set(raw["c161"] + proposals[arm]))
                    decision = mixed.panel.middle.pair.admit(dict(raw, c161=hits), event, echo, zero)
                    row[arm] = dict(proposals=proposals[arm], allowed=decision["allowed_model"], opened=decision["opened"])
                    arms.append(arm)
                if timing in ("original", "external"):
                    lane = "external" if timing == "external" else lanes[name]
                    check_original(row, clock, scores["overlap"], old_plain.pop((*key, lane)), saved)
            for arm in arms:
                total = counts[f"{timing}.{name}.{arm}"]
                mixed.study.tally(total, s, row[arm])
            rows.append(row)
            if len(rows) % 126 == 0:
                print("host", len(rows), flush=True)
        assert len(rows) == 756 and not old_marked and not old_plain
        by_scene = defaultdict(dict)
        for r in rows:
            s = r["source"]
            by_scene[s["index"], s["external_db"], r["timing"]][r["track"]] = r
        for (_, _, timing), tracks in by_scene.items():
            if timing != "external":
                assert tracks["plain_no_address"]["prefix_sha256"] == tracks["panel_music_only"]["prefix_sha256"]
        report = dict(records=rows, summary=counts, model=history["baseline"], overlap_model=ids["overlap"],
                      five_model=five_ids["trained"], cutoff=history["cutoff"], sources={p.name:ch.sha(p) for p in
                      (overlap.OUT, archive, playback.OUT, playback.balanced.PLAY, mixed.panel.demo.WAVE,
                       Path(mixed.__file__), Path(overlap.__file__), Path(__file__))},
                      policy="Four previously consumed voices with three commands and three external levels. "
                      "Keep original music, PCM length, external wet track and preamp within each track; shift "
                      "the panel audio and its TX event together. TX at command -160 ms, -80 ms or 0. "
                      "Plain tracks use original per-scene gain and balanced marked tracks use the fixed "
                      "headroom gain. Only the unmarked tracks also run the overlap proposer and immediate "
                      "name readout. Timely means first delivered OPEN >= address start and < command start.")
        ch.save(work/OUT.name, report)
        (work/OUT.name).replace(OUT)


def guest():
    history = ch.read(OUT)
    reference = ch.read(playback.OUT)
    expected = {(r["source"]["index"], r["source"]["external_db"], r["track"]): (i, r)
                for i, r in enumerate(history["records"]) if r["timing"] == "tx80" and r["track"].startswith("marked")}
    builds = dict(bca=mixed.panel.middle.WORK/"build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == reference["firmware"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == reference["codec"]
    small, sid = ch.load_reader()
    turbo = tx.diag.reader(reference["turbo"])
    proc, model, qid = playback.balanced.original.asr.load()
    assert sid == reference["small"] and qid == reference["qwen"]
    readers = small, turbo, uplink.decoder_api(), proc, model
    rows, counts = [], defaultdict(Counter)
    archive = GUEST.with_suffix(".zip")
    identity = [ch.sha(OUT), ch.sha(playback.OUT), ch.sha(Path(__file__)), reference["firmware"], reference["codec"]]
    run = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(".local/wakeword/c161-connected-panel-timing") / run
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=GUEST.parent) as folder:
        work = Path(folder)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for timing, name, s, mic, _ in scenes():
                if timing != "tx80" or not name.startswith("marked"):
                    continue
                index, host_row = expected.pop((s["index"], s["external_db"], name))
                assert host_row["source"] == s and host_row["pcm_sha256"] == ch.digest(mic)
                role = "both" if name == "marked" else "no_address"
                case = index, s, "bca", role, mic, np.zeros_like(mic), host_row["baseline"]
                idx = len(rows)
                shard = cache / f"{idx:03}.zip"
                if not shard.exists():
                    partial = shard.with_suffix(".tmp")
                    with ZipFile(partial, "w", ZIP_DEFLATED) as capture:
                        r = playback.guest_case(work, capture, idx, case, readers, builds)
                        capture.writestr("case.json", json.dumps(r))
                    partial.replace(shard)
                with ZipFile(shard) as capture:
                    assert capture.testzip() is None
                    r = json.loads(capture.read("case.json"))
                    assert (r["index"], r["host_index"], r["source"], r["role"]) == (idx, index, s, role)
                    for member in capture.namelist():
                        if member != "case.json":
                            logs.writestr(member, capture.read(member))
                assert r["opened"] == host_row["baseline"]["opened"] and r["mic_sha256"] == host_row["pcm_sha256"]
                total = counts[f'{name}.{s["external_db"]}']
                timely = mixed.study.tally(total, s, r)
                total.update(timely_qwen=int(timely and r.get("qwen", {}).get("complete", False)))
                rows.append(r)
                if len(rows) % 12 == 0:
                    print("P4", len(rows), flush=True)
        assert not expected and len(rows) == 72
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(set(logs.namelist())) == len(logs.namelist())
        ch.save(work/GUEST.name, dict(records=rows, summary=counts, host_sha256=ch.sha(OUT),
                model=history["model"], firmware=reference["firmware"], codec=reference["codec"],
                small=sid, turbo=reference["turbo"], qwen=qid, archive_sha256=ch.sha(work/archive.name),
                script_sha256=ch.sha(Path(__file__)),
                policy="All three external levels, TX marker 80 ms before the command, balanced marked track "
                       "and removed-address control. P4 listener and 24 kb/s Opus in QEMU; GPU ASR on full uplink."))
        (work/archive.name).replace(archive)
        (work/GUEST.name).replace(GUEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("host", "guest"))
    (host if parser.parse_args().mode == "host" else guest)()
