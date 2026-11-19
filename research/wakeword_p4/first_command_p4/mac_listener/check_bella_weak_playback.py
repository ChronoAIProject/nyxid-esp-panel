"""Replay every weak af_bella command with paired source mutes and speaker rotation."""
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import replay_bella_command_prefix as bella

ch, tx, study = bella.ch, bella.tx, bella.study
PACK = ch.RESULTS / "c161-bella-command-playback.zip"
OUT = ch.RESULTS / "4090-c161-bella-weak-playback.json"
# Panel TX stays on the published clock in every cell, even when its acoustic signal is muted.
CONDITIONS = (
    ("back", True, True),
    ("back", False, True),
    ("back", True, False),
    ("back", False, False),
    ("front", True, True),
)


def evaluate():
    previous = ch.read(bella.OUT)
    weak = {(r["batch"], r["source"]["index"]): r for r in previous["records"] if r["source"]["external_db"] == -6}
    lengths = {r["source"]["index"]: r["samples"] for r in ch.read(study.timing.OUT)["records"]
               if r["track"] == "marked" and r["timing"] == "tx80" and r["source"]["external_db"] == -6}
    pack_hash = ch.sha(PACK)
    assert pack_hash == "67b54cc6493d1925914d0628e1785ca693e1b41dc5ee61e18c49f48858fe2e94"
    panel, rate = sf.read(study.mixed.panel.demo.WAVE, dtype="int16")
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    filters, detail = bella.gains.direction.filters()
    native = tx.panel.grid.room.rooms.c161_binding()
    records, counts = [], defaultdict(Counter)
    with TemporaryDirectory(dir=OUT.parent) as folder, ZipFile(PACK) as pack:
        matcher = tx.single.timed.binding(Path(folder))
        manifest = json.loads(pack.read("manifest.json"))
        assert rate == 16000 and manifest["report_sha256"] == ch.sha(bella.OUT)
        for entry in manifest["records"]:
            name = Path(entry["file"])
            parent = weak.pop((name.parent.name, int(name.stem)))
            s = parent["source"]
            data = pack.read(entry["file"])
            dry, rate = sf.read(BytesIO(data), dtype="int16")
            assert rate == 16000 and ch.digest(dry) == entry["pcm_sha256"] == s["dry_pcm_sha256"]
            negative = dry.copy()
            negative[:s["command_at"]] = 0
            length = len(dry)
            if parent["batch"] == "first":
                _, (hx, _, _) = tx.diag.room(s["geometry"])
                length = lengths.pop(s["index"]) - 4800 - len(hx) + 1
            group = {"length": (entry, range(length))}
            mix = {play: study.mixed.room_scene(s, group, signal, v5, music, music_id)
                   for play, signal in ((True, panel), (False, np.zeros_like(panel)))}
            for angle, panel_on, music_on in CONDITIONS:
                for role, wave in (("fixed", dry), ("no_address", negative)):
                    # Both angles use the directional filter including the common front SPL reference.
                    external = bella.rear_filter(wave, filters[angle])
                    mic, clips = mix[panel_on](external, include_music=music_on)
                    result = study.playback.host_result(matcher, native, mic, s["transmit_event"])
                    if (angle, panel_on, music_on) == ("back", True, True):
                        if role == "fixed":
                            assert result == parent["result"]
                        else:
                            assert result["pcm_sha256"] == parent["no_address_pcm_sha256"] and not result["opened"]
                    clock = next(iter(result["opened"]), None)
                    timely = clock is not None and s["start"] <= clock < s["command_at"]
                    condition = f"{angle}_panel{int(panel_on)}_music{int(music_on)}"
                    counts[condition+"."+role].update(cases=1, timely=int(timely), opened=int(clock is not None))
                    records.append(dict(file=entry["file"], source=s, condition=condition, role=role,
                        timely=timely, original_target_timely=parent["timely"], clipped_samples=clips, result=result))
        assert not weak and not lengths and len(records) == 240
        ch.save(Path(folder)/OUT.name, dict(records=records, summary=counts, playback_sha256=pack_hash,
                parent_sha256=ch.sha(bella.OUT), model=previous["model"], filters=detail, script_sha256=ch.sha(Path(__file__)),
                policy="Use all 24 weak levels of the same development playback. Change only the indicated acoustic "
                       "source or external angle. Hold the original room, preamp and independent software TX event. "
                       "Compare each cell with its matched no-address control. Host INT8 plus frozen matcher."))
        (Path(folder)/OUT.name).replace(OUT)


if __name__ == "__main__":
    evaluate()
