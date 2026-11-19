"""Verify WakeNet clocks on each panel mixture and score its actual P4-encoded uplink."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import complete_quieter_panel_matrix as panel
import panel_willow_reference as willow
from score_handoff_willow import willow_events
from score_paired_piper_willow import capture

ch, tx = panel.ch, panel.tx
OUT = ch.RESULTS / "4090-c161-quieter-panel-willow.json"


def main():
    plan = ch.read(willow.PLAN)
    plan_hash = ch.sha(willow.PLAN)
    nyx = ch.read(panel.OUT)
    baseline = {r["host_index"]: r for r in nyx["records"]}
    small, identity = ch.load_reader()
    assert identity == nyx["small"]
    turbo = tx.diag.reader(nyx["turbo"])
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == nyx["codec"]
    decoder = uplink.decoder_api()
    scratch = OUT.with_suffix(".tmp")
    rows, summary = [], defaultdict(Counter)
    with TemporaryDirectory() as folder, ZipFile(scratch, "w", ZIP_DEFLATED) as z:
        work = Path(folder)
        uplink.WORK = work
        native = tx.panel.grid.room.rooms.c161_binding()
        matcher = tx.single.timed.binding(work)
        for i, s in enumerate(plan["records"]):
            path = Path(s["path"])
            x, rate = sf.read(path, dtype="int16")
            assert rate == 16000 and ch.sha(path) == s["sha256"] and ch.digest(x) == s["pcm_sha256"]
            old = baseline[s["host_index"]]
            hits = willow_events(i, s, plan_hash, z, runs=willow.WORK / "runs")
            opened = (hits[0]+255)//256*256 if hits else None
            timely = opened is not None and s["start"] <= opened < s["end"]
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, x)
            clock = old["transmit_event"]
            admitted = panel.middle.pair.admit(raw, clock, panel.middle.single.hits(x, clock), np.zeros_like(x))
            if s["target_id"] == "nyx":
                assert (admitted["opened"], raw["pcm_sha256"]) == (old["opened"], old["mic_sha256"])
            row = dict(index=i, host_index=s["host_index"], role=s["target_id"], external_db=s["external_db"],
                       start=s["start"], command_at=s["end"], voice=s["voice"], carrier=s["carrier"],
                       pcm_sha256=s["pcm_sha256"], samples=len(x), hits=hits, opened=opened, timely=timely,
                       nyx_open=admitted["opened"], nyx_c161=raw["c161"], nyx_fixed=raw["fixed"],
                       small=False, turbo=False)
            if opened is not None and s["target_id"].startswith("willow_"):
                cap = capture(small, decoder, work, z, i, "willow", x, opened, s["carrier"])
                log = z.read(f"opus/{i:03}-willow.log").decode().replace("\r\n", "\n")
                samples = cap["emitted"]-cap["first"]
                for lane, key in enumerate(("selected","raw")):
                    packets = uplink.packets(log, i, samples, lane)
                    pcm = uplink.decode(decoder, packets)
                    assert ch.digest(pcm) == cap["lanes"][key]["decoded_sha256"]
                    if lane == 0:
                        reading = uplink.recognition(turbo, pcm, s["carrier"])
                row.update(capture=cap, turbo_result=reading, small=cap["lanes"]["selected"]["asr"]["complete_suffix"],
                           turbo=reading["complete_suffix"])
            summary[f'{s["target_id"]}.{s["external_db"]}'].update(trials=1, opened=int(opened is not None),
                timely=int(timely), small=int(row["small"]), turbo=int(row["turbo"]),
                timely_turbo=int(row["turbo"] and timely), nyx_opens=int(bool(row["nyx_open"])))
            rows.append(row)
            if (i+1)%20 == 0:
                print(i+1, "of", len(plan["records"]), flush=True)
        for entry in sorted(work.glob("*-attempt*.gz")):
            z.write(entry, "retry/"+entry.name)
    assert len(rows) == 240 and len(baseline) == 60
    with ZipFile(scratch) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    archive = OUT.with_suffix(".zip")
    scratch.replace(archive)
    ch.save(OUT, dict(records=rows, summary=summary, fixed_summary=nyx["summary"], archive_sha256=ch.sha(archive),
        small=identity, turbo=nyx["turbo"], firmware=nyx["codec"], nyx_firmware=nyx["firmware"],
        sources={p.name:ch.sha(p) for p in (willow.PLAN, panel.OUT, Path(__file__))},
        policy="Official default WakeNet 9 Willow under ESP32 emulation with verified raw frame returns and "
               "81600-sample prime. Block OPEN, delayed zero-R FIFO and actual P4 Opus match the fixed Nyx route. "
               "No-OPEN counts as incomplete. Also replay each input through the frozen C161 and single-C native "
               "receiver with the panel event. All four roles share one local Nyx demo and one command clock; "
               "Willow gets an ordinary same-voice synthesized address, Nyx uses its fixed cue."))


if __name__ == "__main__":
    main()
