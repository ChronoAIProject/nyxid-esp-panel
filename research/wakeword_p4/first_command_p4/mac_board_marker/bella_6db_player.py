"""Apply a calibrated gain to the Mac target player and keep the opposite source fixed."""
from pathlib import Path
import sys

import bella_surrey_orientations as study

WORK = Path(".local/wakeword/bella-6db-player")
OUT = study.OUT.with_name("4090-bella-6db-player.json")
PLAN = OUT.with_name(OUT.stem + "-plan.json")
PROTO = study.PROTOCOL.with_name("bella_6db_player_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import bella_240_masking as sources

    P = sources.P
    previous = P.read(sources.angles.PLAN)
    gain = 10**(6/20)
    files = {v: previous["files"][v] for v in previous["runtime"].values()}
    for p in (PROTO, SCRIPT, sources.angles.PLAN, sources.pack.OUT, Path(sources.__file__).resolve().relative_to(Path.cwd())):
        files[str(p)] = P.sha(p)
    rows = []
    WORK.mkdir(exist_ok=True, parents=True)
    for old, dry, target, talk, noise in sources.stems_for({120, 240}, ("early-nyx", "headroom", "willow", "absent")):
        bg = talk + noise
        assert P.digest(np.rint(target + bg).clip(-32768, 32767).astype("<i2")) == old["pcm_sha256"]
        # The pack stores mono PCM24 at quarter scale relative to these unpacked values.
        peak = float(np.max(np.abs(dry))*gain / 131072)
        assert peak < 1
        total = np.rint(gain*target + bg)
        pcm = total.clip(-32768, 32767).astype("<i2")
        clipped = total != pcm
        label = old["label"] + "-plus6"
        path = WORK / (label + ".wav")
        sf.write(path, pcm, 16000, subtype="PCM_16")
        files[str(path)] = P.sha(path)
        ref = old["reference_path"]
        files[ref] = previous["files"][ref]
        row = {k: old[k] for k in ("angle", "direction", "scene", "role", "start", "end", "command", "reference_path", "ref_sha256", "samples")}
        rows.append(dict(row, label=label, mode="plus6", gain_db=6, path=str(path), sha256=files[str(path)],
                         pcm_sha256=P.digest(pcm), baseline_label=old["label"], baseline_pcm_sha256=old["pcm_sha256"],
                         dry_peak_full_scale=peak, mic_abs_peak_before_clip=int(np.max(np.abs(total))), clipped=int(clipped.sum()),
                         clipped_before_command=int(clipped[:old["end"]].sum())))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 48
    P.save(PLAN, dict(records=rows, files=files, runtime=previous["runtime"], threshold_requests=["DEFAULT"]))


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage == "prepare":
        prepare()
    elif stage == "case":
        sys.path.insert(0, str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    else:
        study.WORK, study.PLAN, study.OUT = WORK, PLAN, OUT
        {"nyx": study.nyx, "report": study.report}[stage]()
        if stage == "report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
