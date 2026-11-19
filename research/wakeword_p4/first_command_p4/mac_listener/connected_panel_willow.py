"""Run official Willow on same-voice addresses with the exact balanced Nyx command samples."""
from pathlib import Path
import sys

import willow_music_cases as runner

PLAN = runner.ROOT / "results/4090-c161-connected-panel-willow-plan.json"
WORK = Path(".local/wakeword/c161-connected-panel-willow")
RUNNER = Path(__file__).resolve()


def prepare():
    import numpy as np
    import soundfile as sf
    import check_balanced_connected_commands as balanced
    import run_c161_willow_matched_scenes_v1 as launcher

    mixed, tx, ch = balanced.mixed, balanced.tx, balanced.ch
    base = ch.read(balanced.OUT)
    groups = mixed.study.sources()
    ref_plan = ch.read(mixed.study.connected.PLAN)
    runtime = ref_plan["runtime"]
    files = {p:ref_plan["files"][p] for p in runtime.values()}
    for p in (RUNNER, Path(runner.__file__), Path(launcher.__file__), Path(launcher.official.__file__),
              launcher.official.SOURCE, balanced.OUT, mixed.study.connected.PLAN, mixed.study.connected.PACK,
              mixed.panel.demo.WAVE, Path(mixed.__file__), Path(balanced.__file__)):
        files[str(p)] = ch.sha(p)
    assert ch.sha(balanced.PLAY) == base["speakers_sha256"]
    spec, v5, _, markers = tx.study.grid.fit.original()
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    quiet, sr = sf.read(mixed.panel.demo.WAVE,dtype="int16")
    assert sr == 16000
    WORK.mkdir(exist_ok=True)
    rows = []
    for ref in base["records"]:
        s = ref["source"]
        group = groups[s["voice"],s["command"]]
        nyx, x = group["Hey Nyx"]
        willow, w = group["Hey Willow"]
        marked = mixed.render(group,v5,spec,markers)["marked"]
        end, we = nyx["end"], willow["end"]
        before,_ = tx.balanced.match_level(v5,x[end:])
        marked[end:] = mixed.master(before)
        assert ch.digest(marked[end:]) == ref["new_level"]["pcm_sha256"]
        # Preserve all Willow samples before the annotated command. Shift only the leading
        # artificial silence to put its original command boundary on the Nyx clock.
        start = end - (we - willow["start"])
        prefix = np.zeros(end,dtype="<i2")
        copied = min(end,we)
        prefix[-copied:] = w[we-copied:we]
        assert start > 0 and (we <= end or not np.any(w[:we-end]))
        willow_active = tx.balanced.dry_levels(prefix[start:])
        nyx_active = tx.balanced.dry_levels(marked[nyx["start"]:end])
        gain_db = min(nyx_active["active_rms_dbfs"]-willow_active["active_rms_dbfs"],
                      -1.-tx.balanced.dry_levels(prefix)["peak_dbfs"])
        adjusted = np.rint(prefix.astype(float)*10**(gain_db/20))
        assert np.abs(adjusted).max() <= 32767
        roles = [("willow_native",prefix,start),("willow_balanced",adjusted.astype("<i2"),start)]
        vi = mixed.study.connected.VOICES.index(s["voice"])
        ci = mixed.study.connected.CARRIERS.index(s["command"])
        scene = mixed.room_scene(s,group,quiet,v5,bed,bed_sha)
        assert ch.digest(scene(marked)[0]) == ref["mic_sha256"]
        if s["external_db"] == mixed.study.LEVELS[(vi+ci)%3]:
            roles.extend((("nyx_marked",marked[:end],nyx["start"]),("no_address",np.zeros(end,"<i2"),nyx["start"])))
        for role, cue, onset in roles:
            pcm, clipped = scene(np.r_[cue,marked[end:]])
            if role == "no_address":
                assert ch.digest(pcm) == ref["no_address"]["pcm_sha256"]
            path = WORK / f"{len(rows):03}.wav"
            sf.write(path,pcm,16000,subtype="PCM_16")
            files[str(path)] = ch.sha(path)
            levels = None if role == "no_address" else tx.balanced.dry_levels(cue[onset:])
            rows.append(dict(host_index=ref["index"],target_id=role,voice=s["voice"],carrier=s["command"],
                room=s["external_room"],geometry=s["geometry"],external_db=s["external_db"],start=onset,end=end,
                transmit_event=s["transmit_event"],cue_levels=levels,marked_levels=nyx_active,command_pcm_sha256=ch.digest(marked[end:]),
                cue_gain_db=float(gain_db) if role == "willow_balanced" else 0., samples=len(pcm),
                clipped=clipped,pcm_sha256=ch.digest(pcm),path=str(path),sha256=files[str(path)]))
    assert len(rows) == 96
    plan = dict(records=rows,files=files,runtime=runtime,threshold_requests=["DEFAULT"],
        policy="Whole-sentence Kokoro Willow prefix at native level or matched to the same voice's marked Nyx "
        "prefix active RMS with -1 dBFS peak cap. For paired command ASR use the exact Nyx command suffix "
        "on both. Preserve the -6 dB local panel, music, room, gain and command clock. One preassigned "
        "gain per family also has Nyx and command-only reciprocal controls. No new independent FAR time.")
    if PLAN.exists():
        assert plan == ch.read(PLAN)
    else:
        ch.save(PLAN,plan)


if __name__ == "__main__":
    runner.PLAN, runner.WORK, runner.RUNNER = PLAN, WORK, RUNNER
    {"prepare":prepare, "run":runner.run, "case":runner.case}[sys.argv.pop(1)]()
