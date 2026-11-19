"""Replay the frozen per-voice selections and equal-level defaults on paired command scenes."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve

import select_room_spoken_cues as selection

ch, room, gap = selection.ch, selection.room, selection.gap
OUT = ch.RESULTS / (selection.STEM + "-replay.json")


def main():
    fit = ch.read(selection.FIT)
    assert fit["pool_sha256"] == ch.sha(selection.PACK)
    choices = fit["choices"]
    top = max(choices, key=lambda v: choices[v]["selected"]["rank"])
    cues = dict((r["index"], (r, x)) for r, x in selection.pool())
    spec = ch.read(gap.study.voices.gate.fit.COEFF)
    costs = ch.read(gap.study.voices.timing.names.OUT)
    net = gap.study.voices.timing.spoken.natural.Detector()
    assert spec["models"] == net.identity and spec["models"]["c161"] == fit["model_sha256"]
    fw = gap.uplink.firmware()
    reader, asr = ch.load_reader()
    decoder = gap.uplink.decoder_api()
    records, captures = [], []
    stats = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        fixed = gap.device.host.fixed.ch.binding(work/"v5.so", coded="v5")
        for s, old, kind, h, bg, cmd in room.room_inputs():
            row = dict(room=s["room"], voice=s["voice"], command=s["carrier"], command_at=s["end"],
                       silence_sha256=s["pcm_sha256"], official_willow=old["official_willow_events"], cues={})
            for v, configs in choices.items():
                for config in ("default", "selected"):
                    for role in ("target", "nick"):
                        idx = configs[config][role]
                        meta, cue = cues[idx]
                        assert meta["voice"] == v and meta["name"] == selection.NAMES[role == "nick"]
                        end = s["end"]
                        at = end-len(cue)
                        if idx not in row["cues"]:
                            wet = fftconvolve(cue.astype(float), h)
                            scene = cmd + bg
                            scene[at:at+len(wet)] += wet
                            mic = room.quantize(scene*s["gain"])
                            trace = net.score(mic)
                            result, _, _ = gap.study.voices.score(net, mic, trace["c161"], trace["proposals"],
                                                               trace["events"], spec, costs)
                            entry = dict(index=idx, cue_at=at, pcm_sha256=ch.digest(mic), c161=trace["c161"], routes={})
                            starts = {}
                            for route in ("c161", "linear"):
                                ts = result[route]
                                events = [t for t in ts if at <= t < len(mic)]
                                rec = dict(events=ts, hits=events)
                                if events:
                                    opened = events[0]
                                    rec.update(timely=opened < end, lead_samples=end-opened)
                                    if role == "target":
                                        if opened not in starts:
                                            i = len(captures)
                                            cap = gap.replay.capture(reader, decoder, work, logs, i, "select",
                                                                     mic, opened, s["carrier"])
                                            captures.append(dict(id=i, scene=len(records), cue=idx, opened=opened, **cap))
                                            starts[opened] = i
                                        i = starts[opened]
                                        rec.update(capture=i, complete=captures[i]["lanes"]["selected"]["asr"]["complete_suffix"])
                                entry["routes"][route] = rec
                            row["cues"][idx] = entry
                        entry = row["cues"][idx]
                        if v == top and config == "selected" and "listener" not in entry:
                            # Rebuild the selected PCM when the default was the same source.
                            signal = cmd + bg
                            wet = fftconvolve(cue.astype(float), h)
                            signal[at:at+len(wet)] += wet
                            mic = room.quantize(signal*s["gain"])
                            assert ch.digest(mic) == entry["pcm_sha256"]
                            pair = np.column_stack((mic, np.zeros_like(mic)))
                            i = 2*len(records) + int(role == "nick")
                            path = work/f"listener-{i:02}.log"
                            gap.uplink.boot(gap.uplink.BUILD, path, b"LISTENER PASS",
                                struct.pack("<3I", 0x4452594e, i, len(mic)) + pair.T.copy().tobytes(), work)
                            detail, lanes = gap.uplink.analyze(path.read_text(),
                                dict(index=i, group="external", command_at=end), pair, fixed)
                            assert detail["c161"] == entry["c161"] and not detail["v5"]
                            events = entry["routes"]["c161"]["events"]
                            assert detail.get("delivered") == next(iter(events), None)
                            capture = entry["routes"]["c161"].get("capture")
                            if capture is not None:
                                for lane in ("raw","selected"):
                                    assert ch.digest(lanes[lane]) == captures[capture]["lanes"][lane]["input_sha256"]
                            entry["listener"] = dict(result=detail, uart=path.name, uart_sha256=ch.sha(path))
                            logs.write(path, path.name)
                            path.unlink()
                        for route, data in entry["routes"].items():
                            stats[config][role][route].update(trials=1, hits=bool(data["hits"]),
                                timely=data.get("timely",False), complete=data.get("complete",False),
                                timely_complete=data.get("timely",False) and data.get("complete",False))
            records.append(row)
            print(kind, len(records), len(captures), flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for p in retries:
            logs.write(p, "retry/"+p.name)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for row in records:
            for entry in row["cues"].values():
                if "listener" in entry:
                    d = entry["listener"]
                    assert hashlib.sha256(logs.read(d["uart"])).hexdigest() == d["uart_sha256"]
        for c in captures:
            i = c["id"]
            text = logs.read(f"opus/{i:03}-select.log").decode().replace("\r\n","\n")
            for lane, key in enumerate(("selected","raw")):
                packets = gap.uplink.packets(text, i, c["emitted"]-c["first"], lane)
                assert ch.digest(gap.uplink.decode(decoder, packets)) == c["lanes"][key]["decoded_sha256"]
    assert len(records) == 16
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=records, captures=captures, summary=stats, training_top_voice=top,
        sources={p.name:ch.sha(p) for p in (selection.FIT, selection.PACK, room.parent.OUT, room.rooms.PLAN,
                gap.study.voices.gate.fit.COEFF, gap.study.voices.timing.names.OUT, Path(__file__))},
        archive_sha256=ch.sha(archive), asr=asr, firmware=fw, retries=[p.name for p in retries],
        policy="Replay 8 voices against each of the same 16 kitchen and living-room clock/command cells. "
        "Compare original context-0/speed-1 baseline vs training-selected index; equal matched active levels and "
        "200 ms tail. Native C161; integer early research route with prior P4 service P95. Full P4 listener "
        "for the highest-ranked training pair only, P4 Opus for each distinct target opening and re-decode. "
        "Scenes are repeated development inputs; synthesis settings and top voice selected before this replay."))


if __name__ == "__main__":
    main()
