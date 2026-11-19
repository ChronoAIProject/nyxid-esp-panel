"""Transfer the connected-speech microphone gain to the existing 11-song fixed-cue room grid."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_connected_mic_headroom as headroom
import check_panel_music_transfer as music

ch, mixed, tx = headroom.ch, headroom.mixed, headroom.tx
OUT = ch.RESULTS / "4090-c161-mic-headroom-music-transfer.json"


def main():
    plan = ch.read(headroom.OUT)
    gain = plan["fixed_preamp_gain"]
    historical = ch.read(music.OUT)
    assert plan["model"] == historical["model"]
    prior = {(r["scene"], r["role"]): r for r in historical["records"] if r["panel"] == "quiet"}
    marker = tx.event_time()
    records, stats = [], defaultdict(Counter)
    with TemporaryDirectory() as folder:
        matcher = tx.single.timed.binding(Path(folder))
        native = tx.panel.grid.room.rooms.c161_binding()
        for index, (s, views) in enumerate(music.room_mixes()):
            clock = s["panel_at"] + marker
            for role in ("both", "no_address"):
                old = prior.pop((index, role))
                floating = views["quiet", role]
                original, _ = mixed.digitize(floating, s["gain"])
                assert old["source"] == s and ch.digest(original) == old["result"]["pcm_sha256"]
                pcm, clipped = mixed.digitize(floating, gain)
                raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, pcm)
                hits = mixed.panel.middle.single.hits(pcm, clock)
                result = mixed.panel.middle.pair.admit(raw, clock, hits, np.zeros_like(pcm))
                tally = stats[f'{role}.{s["external_db"]}']
                timely = mixed.study.tally(tally, s, result)
                before = old["timely"]
                tally.update(previous_timely=int(before), gained=int(timely and not before), lost=int(before and not timely),
                             same_clock=int(result["opened"] == old["result"]["opened"]), clipped=clipped,
                             saturated_cases=int(clipped > 0))
                records.append(dict(scene=index, source=s, role=role, previous_opened=old["result"]["opened"],
                                    previous_timely=before, timely=timely, result=result, clipped_samples=clipped,
                                    peak_dbfs=headroom.peak_dbfs(floating * gain)))
            if (index + 1) % 30 == 0:
                print("Room pairs", index + 1, flush=True)
    assert not prior and len(records) == 330 and all(c["trials"] == 55 for c in stats.values())
    report = dict(records=records, summary=stats, fixed_preamp_gain=gain, model=plan["model"],
                  sources={p.name:ch.sha(p) for p in (headroom.OUT, music.OUT, Path(music.__file__),
                           Path(headroom.__file__), Path(mixed.__file__), Path(__file__))},
                  policy="Use the fixed gain from the connected-speech experiment, without fitting to this grid. "
                         "Quiet panel, fixed v5 with separately synthesized commands, 11 earlier development songs, "
                         "22 room responses, five offsets, three external levels. Host listener only. Previous "
                         "source.gain describes the historical render. No new independent background hours.")
    tmp = OUT.with_suffix(".tmp")
    ch.save(tmp, report)
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
