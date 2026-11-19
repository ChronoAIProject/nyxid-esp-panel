"""Compare first commands from the repeated cue on the established room inputs and archived P4 streams."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
from scipy.signal import fftconvolve

import check_repeat_p4 as repeat_p4
import compare_room_willow_uplink as prior

ch, room, gap = prior.ch, prior.room, prior.gap
repeat = repeat_p4.repeat
OUT = ch.RESULTS / (repeat.source.STEM + "-commands.json")


def decoded(logs, path, decoder, index, samples, lane, expected_sha):
    log = logs.read(path).decode().replace("\r\n", "\n")
    packets = gap.uplink.packets(log, index, samples, lane)
    pcm = gap.uplink.decode(decoder, packets)
    assert ch.digest(pcm) == expected_sha
    return pcm, len(packets)


def main():
    benchmark = ch.read(prior.OUT)
    subset = ch.read(repeat_p4.OUT)
    reader, small = ch.load_reader()
    turbo_info = benchmark["turbo"]
    assert small == benchmark["small"] == subset["asr"]
    model_dir = prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n, h in turbo_info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = gap.uplink.decoder_api()
    totals = defaultdict(lambda: defaultdict(Counter))

    def measure(cohort, name, opened, start, end, small_complete, turbo_complete):
        timely = opened is not None and start <= opened < end
        totals[cohort][name].update(trials=1, timely=timely, small_complete=small_complete,
                                    turbo_complete=turbo_complete, turbo_timely_complete=timely and turbo_complete)

    six = []
    p4_zip = repeat_p4.OUT.with_suffix(".zip")
    assert subset["archive_sha256"] == ch.sha(p4_zip)
    with ZipFile(p4_zip) as logs:
        for row in subset["records"]:
            if row["local"]:
                continue
            result = row["result"]
            source = result["source"]
            opened = next(iter(result["opened"]), None)
            entry = dict(case=row["case"], cue=row["cue"], room=row["room"], command=row["command"], opened=opened)
            small_done = turbo_done = False
            if opened is not None:
                cap = result["uplink"]
                i = row["case"]
                pcm, packets = decoded(logs, f"{i:02}-opus.log", decoder, i, cap["samples"], 0, cap["pcm_sha256"])
                score = gap.uplink.recognition(turbo, pcm, row["command"])
                small_done = cap["recognition"]["complete_suffix"]
                turbo_done = score["complete_suffix"]
                entry.update(packets=packets, pcm_sha256=ch.digest(pcm), small=cap["recognition"], turbo=score)
            measure("six_room_music", row["cue"], opened, source["cue_at"], source["command_at"], small_done, turbo_done)
            six.append(entry)

    original = repeat.grid.fit.original()[1]
    trained = repeat.tracks()["repeat_trained"]
    assert len(original) == len(trained)
    native = room.rooms.c161_binding()
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    scenes = []
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        detector = ch.binding(work / "cue.so", coded="v5")
        for i, ((s, old, kind, h, bg, cmd), reference) in enumerate(zip(
                room.room_inputs(), benchmark["records"], strict=True)):
            end = s["end"]
            start = end - len(original)
            assert (s["voice"], s["carrier"], end, kind, i) == (
                reference["voice"], reference["command"], reference["command_at"], reference["kind"], reference["scene"])
            scene = dict(voice=s["voice"], command=s["carrier"], kind=kind, start=start, end=end,
                         silence_sha256=s["pcm_sha256"], rir_sha256=old["rir_sha256"], routes={})
            for name, cue in (("fixed_v5", original), ("repeat_trained", trained)):
                mixed = bg + cmd
                wet = fftconvolve(cue.astype(float), h)
                mixed[start:start+len(wet)] += wet
                mic = room.quantize(mixed*s["gain"])
                route = repeat.grid.outcome(detector, native, mic, dict(start=start, command_at=end))
                opened = route["opened"]
                if name == "fixed_v5":
                    assert route["pcm_sha256"] == old["routes"][name]["pcm_sha256"]
                    assert route["opened"] == reference["routes"][name]["opened"]
                    assert route["fixed"] == old["routes"][name]["fixed"] and route["c161"] == old["routes"][name]["c161"]
                    route["scores"] = reference["routes"][name]["full"]
                elif opened is not None:
                    route["capture"] = gap.replay.capture(reader, decoder, work, logs, i, "repeat", mic, opened, s["carrier"])
                scene["routes"][name] = route
            scenes.append(scene)
            print(i, kind, flush=True)
        for retry in sorted(work.glob("*-attempt*.gz")):
            logs.write(retry, "retry/" + retry.name)
    with ZipFile(tmp) as logs:
        assert logs.testzip() is None
        for i, scene in enumerate(scenes):
            for name, route in scene["routes"].items():
                if "capture" in route:
                    cap = route["capture"]
                    for lane, label in enumerate(("selected", "raw")):
                        lane_info = cap["lanes"][label]
                        pcm, count = decoded(logs, f"opus/{i:03}-repeat.log", decoder, i,
                                             cap["emitted"]-cap["first"], lane, lane_info["decoded_sha256"])
                        assert count == lane_info["packets"]
                        if lane == 0:
                            route["scores"] = dict(small=lane_info["asr"],
                                                   turbo=gap.uplink.recognition(turbo, pcm, scene["command"]))
                scores = route.get("scores")
                measure(scene["kind"], name, route["opened"], scene["start"], scene["end"],
                        bool(scores and scores["small"]["complete_suffix"]),
                        bool(scores and scores["turbo"]["complete_suffix"]))
    assert len(six) == 18 and len(scenes) == 16
    tmp.replace(archive)
    ch.save(OUT, dict(six_rooms=six, willow_scenes=scenes, summary=totals, small=small, turbo=turbo_info,
                      archive_sha256=ch.sha(archive), firmware=gap.uplink.firmware(),
                      sources={p.name: ch.sha(p) for p in (prior.OUT, repeat_p4.OUT, p4_zip, room.parent.OUT,
                               repeat.source.PLAN, Path(__file__))},
                      policy="Six music rooms use the archived guest FIFO and P4 Opus. The 16 established "
                      "kitchen/living-room inputs use the matched Willow benchmark's host OPEN clock, two-second "
                      "pre-roll and P4 Opus. Compare complete final command words in full selected-lane captures "
                      "using the two unprompted readers. Reused rooms and voices."))


if __name__ == "__main__":
    main()
