"""Check the Mac room handoff with both Nyx receiver policies in P4 QEMU."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

import check_fixed_playback_only as fixed
import package_bella_false_wake_room as case072

room, P = case072.room, case072.P
OUT = room.trial.RESULTS / "4090-c161-bella-room-handoff-replay.json"


def inputs():
    rows = []
    for report in (room.OUT, case072.REPORT):
        handoff = P.read(report)
        pack = room.trial.WORK / handoff["archive"]
        assert P.sha(pack) == handoff["archive_sha256"]
        with ZipFile(pack) as z:
            manifest = json.loads(z.read("manifest.json"))
            if report == room.OUT:
                entries = [(p, s["reference_sha256"], s["command_at"])
                           for s in manifest["scenes"] for p in s["pairs"]]
            else:
                entries = [(r, r["reference_sha256"], manifest["command_at"]) for r in manifest["records"]]
            for item, ref_hash, at in entries:
                pcm, rate = sf.read(BytesIO(z.read(item["digital"])), dtype="int16", always_2d=True)
                assert rate == 16000 and pcm.shape[1] == 2
                mic, ref = (pcm[:, ch].astype("<i2").tobytes() for ch in (0, 1))
                assert sha256(mic).hexdigest() == item["mic_sha256"] and sha256(ref).hexdigest() == ref_hash
                label = item["digital"].removeprefix("digital/").removesuffix(".wav")
                rows.append((label, mic, ref, None, at))
    assert len(rows) == len({r[0] for r in rows}) == 35
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python", type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__), room.OUT, case072.REPORT, fixed.OUT, fixed.PATCH,
               room.daily.OUT, room.trial.WAKE, case072.replay.OUT]
    known = P.read(fixed.OUT)
    baseline = case072.replay.stage.GUEST
    assert known["model_sha256"] == P.sha(fixed.base.MODEL)
    assert P.sha(fixed.QEMU) == known["qemu_sha256"] and P.sha(fixed.PATCH) == known["sources"][fixed.PATCH.name]
    prior = {}
    for report in (room.daily.OUT, room.trial.WAKE, case072.replay.OUT):
        data = P.read(report)
        assert data["guest_sha256"] == P.sha(baseline) and P.sha(report.with_suffix(".zip")) == data["archive_sha256"]
        for row in data["records"]:
            prior[row["input_sha256"]] = (report.with_suffix(".zip"), row)

    cases = inputs()
    records = {c[0]: dict(command_at=c[-1]) for c in cases}
    with TemporaryDirectory(dir=room.trial.WORK) as folder:
        work = Path(folder).resolve()
        cue_only = fixed.build(work, args.idf_python)
        trace = OUT.with_suffix(".zip")
        with ZipFile(work / trace.name, "w") as logs:
            fresh = []
            for c in cases:
                label, mic, ref, _, at = c
                key = sha256(fixed.run_guest.payload(mic, ref)).hexdigest()
                records[label]["input_sha256"] = key
                if key not in prior:
                    fresh.append(c)
                    continue
                archive, old = prior[key]
                assert old["command_at"] == at
                with ZipFile(archive) as z:
                    raw = z.read(old["label"] + ".log")
                assert sha256(raw).hexdigest() == old["uart_sha256"]
                room.daily.matched.member(logs, "baseline/" + label + ".log", raw)
                records[label]["baseline"] = dict(events=old["events"], uart_sha256=old["uart_sha256"],
                                                  from_archive=archive.name)

            for policy, guest, subset in (("baseline", baseline, fresh), ("cue_only", cue_only, cases)):
                replay = fixed.run_guest.replay_cases(guest, fixed.SIM, fixed.QEMU, work / policy, subset)
                for observed in replay["records"]:
                    label, key = observed["label"], observed["input_sha256"]
                    assert records[label]["input_sha256"] == key
                    raw = (work / policy / "uart" / (key + ".log")).read_bytes()
                    assert sha256(raw).hexdigest() == observed["uart_sha256"]
                    room.daily.matched.member(logs, policy + "/" + label + ".log", raw)
                    records[label][policy] = dict(events=observed["events"], uart_sha256=observed["uart_sha256"])
                print(policy, len(subset), "digital P4 replays", flush=True)

        assert len(fresh) == 9 and all(c[0].endswith("/willow") for c in fresh)
        with ZipFile(work / trace.name) as z:
            assert z.testzip() is None and len(z.namelist()) == 70
        P.save(work / OUT.name, dict(records=records, archive_sha256=P.sha(work / trace.name),
               sources={p.name:P.sha(p) for p in sources}, baseline_guest_sha256=P.sha(baseline),
               cue_only_guest_sha256=P.sha(cue_only), qemu_sha256=P.sha(fixed.QEMU), model_sha256=known["model_sha256"],
               policy="Three weak scenes with Nyx, Willow and erased addresses at three gain settings, "
               "plus the eight source-mute settings of case 072. The Willow audio is a reciprocal input "
               "to the Nyx receiver. Both policies use the same C161 and fixed marker. Prior baseline "
               "UARTs are reused only on identical serialized P4 inputs."))
        (work / trace.name).replace(trace)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
