"""Prepare and execute the official WakeNet reference on the quieter-panel command matrix."""
from pathlib import Path
import sys

import willow_music_cases as runner

PLAN = runner.ROOT / "results/4090-c161-quieter-panel-willow-plan.json"
WORK = Path(".local/wakeword/c161-quieter-panel-willow")
RUNNER = Path(__file__).resolve()


def prepare():
    from io import BytesIO
    from zipfile import ZipFile

    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve

    import complete_quieter_panel_matrix as bench
    import run_c161_willow_matched_scenes_v1 as official

    ch, tx = bench.ch, bench.tx
    WORK.mkdir(exist_ok=True)
    known = ch.read(bench.OUT)
    parent = {r["host_index"]: r for r in known["records"]}
    dry = tx.study.grid.room.rooms.DRY
    runtime_plan = ch.read(tx.study.grid.room.rooms.PLAN)
    runtime = runtime_plan["runtime"]
    assert ch.sha(dry) == runtime_plan["files"][str(dry)]
    files = {p: runtime_plan["files"][p] for p in runtime.values()}
    for path in (RUNNER, Path(runner.__file__), Path(official.__file__), Path(official.official.__file__),
                 official.official.SOURCE, dry, bench.OUT, bench.demo.WAVE, tx.diag.BOARD):
        files[str(path)] = ch.sha(path)
    v5, _ = tx.study.panel_wave()
    demo, rate = sf.read(bench.demo.WAVE, dtype="int16")
    assert rate == 16000
    rows = []
    with ZipFile(dry) as pack:
        for src in ch.read(tx.diag.BOARD)["records"]:
            if (src["name"], src["role"]) != ("bca", "both"):
                continue
            host_index, scene = src["host_index"], src["source"]
            assert parent[host_index]["source"] == scene
            _, bg, _, cmd, h, ext_gain, _ = tx.diag.scene(src)
            (hp, _, _), _ = tx.diag.room(scene["geometry"])
            command, _ = tx.balanced.match_level(v5, cmd)
            common = bg.copy()
            local = fftconvolve(demo.astype(float), hp)
            local_at, end = scene["local_at"], scene["command_at"]
            common[local_at:local_at+len(local)] += local
            w, rate = sf.read(BytesIO(pack.read(scene["voice"] + "-Willow.wav")), dtype="int16")
            assert rate == 16000
            target = np.mean(v5.astype(float)**2)
            gain = np.sqrt(target / np.mean(w.astype(float)**2))
            louder = np.rint(w.astype(float) * gain)
            assert np.abs(louder).max() <= 32767
            for role, cue in (("nyx", v5), ("willow_native", w), ("willow_rms", louder.astype("<i2")),
                              ("no_address", np.zeros_like(v5))):
                start = end-len(cue)
                external = fftconvolve(np.r_[cue, command].astype(float), h) * ext_gain
                mic = common.copy()
                mic[start:start+len(external)] += external
                floating = mic * scene["gain"]
                pcm = tx.study.grid.room.quantize(floating)
                if role == "nyx":
                    assert ch.digest(pcm) == parent[host_index]["mic_sha256"]
                path = WORK / f"{len(rows):03}.wav"
                sf.write(path, pcm, 16000, subtype="PCM_16")
                files[str(path)] = ch.sha(path)
                rows.append(dict(host_index=host_index, target_id=role, room=scene["external_room"],
                    geometry=scene["geometry"], voice=scene["voice"], carrier=scene["command"],
                    panel_at=local_at, start=start, end=end, external_db=scene["external_db"],
                    cue_sha256=ch.digest(cue), command_pcm_sha256=ch.digest(command),
                    cue_gain=float(gain) if role=="willow_rms" else 1., samples=len(pcm),
                    clipped=int(np.count_nonzero((np.rint(floating) < -32768) | (np.rint(floating) > 32767))),
                    pcm_sha256=ch.digest(pcm), path=str(path), sha256=files[str(path)]))
    assert len(rows) == 240
    result = dict(runtime=runtime, records=rows, threshold_requests=["DEFAULT"], files=files,
        policy="External address ends at the same command onset in all four views. Native Willow and "
               "dry-RMS-matched Willow use the command speaker, Nyx uses the original fixed v5 waveform. "
               "All retain the identical quieter local Nyx demonstration, balanced command, room, gain and music. "
               "Include full external Nyx and command-only controls for the official receiver.")
    if PLAN.exists():
        assert ch.read(PLAN) == result
    else:
        ch.save(PLAN, result)


if __name__ == "__main__":
    runner.PLAN, runner.WORK, runner.RUNNER = PLAN, WORK, RUNNER
    {"prepare": prepare, "run": runner.run, "case": runner.case}[sys.argv.pop(1)]()
