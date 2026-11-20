"""Compare the two shipped rev1 listener implementations on registered digital M/R pairs."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import bella_reference_barge as dual
import check_address_gain_wakes as wakes
import check_bella_early_groups_3db as shifts
import check_bella_plus2_playback as playback
import eight_local_player_079 as player

fixed, P = wakes.fixed, dual.P
WORK = Path(".local/wakeword/eight-local-fixed-only")
OUT = player.OUT.with_name("4090-eight-local-fixed-only.json")
PLAN = player.ROOT / "experiments/eight_local_fixed_only_protocol.md"
ROOM077 = OUT.with_name("4090-early-groups-eight-local-077.json")
PLAN077 = ROOM077.with_stem(ROOM077.stem + "-plan")
DIAG = shifts.OUT.with_stem(shifts.OUT.stem.replace("shifts", "diagnosis"))
COLLISIONS = DIAG.with_stem(DIAG.stem.replace("diagnosis", "collisions"))
PANEL = shifts.candidate.REPLAY
DAILY = {"six-cups", "command1-talk1-music1"}
TWICE = {"079/pcm48-new-af_bella", "079/aac48-erased-erased", "source/210-front-079-voice-None",
         "panel/3/synchronous"} | {"daily/" + label for label in DAILY}


def indexed(path):
    rows = P.read(path)["records"]
    result = {row["label"]: row for row in rows}
    assert len(rows) == len(result)
    return result


def channels(row):
    mic, ref = (sf.read(row[key], dtype="int16") for key in ("path", "reference_path"))
    assert mic[1] == ref[1] == 16000
    assert P.sha(Path(row["path"])) == row["sha256"]
    assert P.digest(mic[0]) == row["pcm_sha256"] and P.digest(ref[0]) == row["ref_sha256"]
    return mic[0], ref[0]


def collision_inputs(cue):
    previous = indexed(COLLISIONS)
    for row in P.read(playback.selfplay.OUT)["records"]:
        if (row["reference"], row["version"]) != ("synchronous", "original"):
            continue
        geometry = row["geometry"]
        (ir, _, checksum), _ = playback.daily.tx.diag.room(geometry)
        assert checksum == row["room_sha256"]
        response = fftconvolve(cue.astype(float), ir) * row["scale"]
        for offset in (-800, -400, 0, 400, 800):
            ext_start = 32000 + offset * 16
            mix = np.zeros(64000 + len(response) + max(0, offset * 16))
            mix[32000:32000 + len(response)] = response
            mix[ext_start:ext_start + len(response)] += response
            mic = np.rint(mix).clip(-32768, 32767).astype("<i2")
            ref = np.zeros_like(mic)
            ref[32000:32000 + len(cue)] = cue
            key = f"{geometry}/{offset}"
            old = previous["candidate/" + key]
            yield "collision/" + key, mic, ref, old, {"geometry": geometry, "offset_ms": offset,
                  "panel_alone": old["panel_alone"], "external_alone": old["external_alone"]}


def cases():
    for row in indexed(player.OUT).values():
        mic, ref = channels(row)
        yield "079/" + row["label"], mic, ref, dict(row, events=row["nyx"]), {
            key: row[key] for key in ("role", "voice", "mode", "start", "end")}
    prior = indexed(ROOM077)
    assert P.sha(PLAN077) == P.read(ROOM077)["plan_sha256"]
    for label, row in indexed(PLAN077).items():
        mic, ref = channels(row)
        old = prior.pop(label)
        assert old["pcm_sha256"] == row["pcm_sha256"]
        yield "077/" + label, mic, ref, old, {key: row[key] for key in ("voice", "mode", "start", "end")}
    assert not prior
    prior = indexed(DIAG)
    for row, role, offset, mic, *_ in shifts.inputs("diagnosis"):
        label = f"{row['angle']}-{row['direction']}-{row['scene']:03}-{role}-{offset}"
        old = prior.pop(label)
        assert P.digest(mic) == old["pcm_sha256"]
        yield "source/" + label, mic, np.zeros_like(mic), old, {
            key: old[key] for key in ("role", "offset_ms", "start", "end")}
    assert not prior
    previous = indexed(fixed.REPORT)
    for label, mic, ref, _, _, _ in fixed.inputs():
        if label in DAILY:
            old = previous[label]
            yield "daily/" + label, np.frombuffer(mic, "<i2"), np.frombuffer(ref, "<i2"), {
                "input_sha256": old["serialized_input_sha256"], "events": old["baseline"],
                "fixed_events": old["fixed"]["epochs"][0]["events"]}, {}
    original, rate = sf.read(playback.daily.board.bella.CUE, dtype="int16")
    cue, cue_rate = sf.read(shifts.candidate.WAV, dtype="int16")
    assert rate == cue_rate == 16000
    assert P.sha(shifts.candidate.WAV) == P.read(shifts.candidate.FIT)["voices"][shifts.candidate.NAME]
    previous = indexed(PANEL)
    for label, row, mic, ref in playback.panel(original, cue):
        yield label, mic, ref, previous[label], {"geometry": row["geometry"], "reference": row["role"]}
    yield from collision_inputs(cue)


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    standalone = P.read(fixed.REPORT)
    build = P.read(dual.WORK / "build.json")
    for path in (player.OUT, ROOM077, DIAG, COLLISIONS, PANEL):
        assert P.read(path)["build"] == build
    assert P.sha(fixed.MODEL) == build["model_sha256"]
    assert P.sha(fixed.QEMU) == standalone["qemu_sha256"] == build["qemu_sha256"]
    images = {"fixed": wakes.fixed_image(standalone, "fixed"),
              "c161": bytearray((dual.WORK / "guest.flash").read_bytes())}
    assert fixed.digest(images["c161"]) == build["flash_sha256"]
    identities = {name: fixed.digest(image) for name, image in images.items()}
    caches = {name: WORK / build["qemu_sha256"] / key for name, key in identities.items()}
    for cache in caches.values():
        cache.mkdir(parents=True, exist_ok=True)
    rows, files = [], {}
    with TemporaryDirectory(dir=WORK) as folder:
        stage = Path(folder)
        archive = OUT.with_suffix(".zip")
        with ZipFile(stage / archive.name, "w", ZIP_DEFLATED) as logs:
            for label, mic, ref, old, meta in cases():
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                key = fixed.digest(block)
                if "input_sha256" in old:
                    assert key == old["input_sha256"], label
                row = dict(meta, label=label, samples=len(mic), mic_sha256=P.digest(mic),
                           ref_sha256=P.digest(ref), input_sha256=key, previous_events=old["events"])
                for name, image in images.items():
                    epochs = 2 if label in TWICE else 1
                    cache = caches[name]
                    for attempt in range(2):
                        try:
                            obs, log, faults = fixed.boot(cache, image, cache, block, epochs)
                            break
                        except AssertionError:
                            if attempt:
                                raise
                    for epoch in obs:
                        events = epoch["events"]
                        if name == "c161":
                            assert events == old["events"], (label, events, old["events"])
                        else:
                            assert not events["M"]
                            if "fixed_events" in old:
                                assert events == old["fixed_events"]
                        assert events == obs[0]["events"]
                    for path in (log, *faults):
                        suffix = "wake.log" if path == log else path.name
                        member = f"{name}/{label}/{suffix}"
                        logs.write(path, member)
                        files[member] = P.sha(path)
                    row[name] = obs
                rows.append(row)
                print(len(rows), label, row["fixed"][0]["events"], row["c161"][0]["events"], flush=True)
        assert len({r["label"] for r in rows}) == len(rows) == 100
        assert Counter(r["label"].split("/")[0] for r in rows) == {
            "079": 36, "077": 24, "source": 10, "daily": 2, "panel": 8, "collision": 20}
        assert {r["label"] for r in rows if len(r["fixed"]) == 2} == TWICE
        with ZipFile(stage / archive.name) as logs:
            assert len(logs.namelist()) == len(files) and not logs.testzip()
            for member, sha in files.items():
                assert fixed.digest(logs.read(member)) == sha
        deps = (Path(__file__), PLAN, player.OUT, ROOM077, PLAN077, DIAG, COLLISIONS, PANEL,
                fixed.REPORT, Path(fixed.__file__), shifts.candidate.FIT)
        P.save(stage / OUT.name, dict(records=rows, files=files, archive_sha256=P.sha(stage / archive.name),
               qemu_sha256=build["qemu_sha256"], flash_sha256=identities, model_sha256=build["model_sha256"],
               guest_sha256=dict(fixed=standalone["guest_sha256"]["fixed"], c161=build["guest_sha256"]),
               sources={path.name: P.sha(path) for path in deps}, independent_background_hours=0))
        (stage / archive.name).replace(archive)
        (stage / OUT.name).replace(OUT)


if __name__ == "__main__":
    run()
