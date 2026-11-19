"""Compare the official Willow DEFAULT and C161 on the handoff's ongoing panel speech."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
BASE = ROOT / "results/4090-c161-handoff-willow"
PLAN = BASE.with_name(BASE.name + "-plan.json")
WORK = Path(".local/wakeword/c161-handoff-willow")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import coded_close_join as joined
    import handoff_barge_in as handoff

    history = joined.read(handoff.BASE.with_suffix(".json"))
    old = joined.read(joined.PLAN)
    rows = { (r["room"], r["voice"], r["carrier"], r["target_id"]): r for r in old["records"] }
    files = {p: old["files"][p] for p in old["runtime"].values()}
    for path in (RUNNER, Path(handoff.__file__), joined.PLAN, handoff.BASE.with_suffix(".json"),
                 joined.matched.official.official.SOURCE, Path(joined.matched.official.official.__file__)):
        files[str(path)] = joined.sha(path)
    WORK.mkdir(exist_ok=True)
    records = []
    for (source, fixed_mic, player, response), result in zip(handoff.scenes(), history["records"], strict=True):
        assert source == result["source"]
        key = (source["room"], source["voice"], source["carrier"])
        for role in ("fixed", "ordinary", "willow", "silence"):
            if role == "fixed":
                dry = fixed_mic
                start = source["cue_at"]
            else:
                prior = rows[(*key, role)]
                path = Path(prior["path"])
                assert joined.sha(path) == prior["sha256"]
                dry, rate = sf.read(path, dtype="int16")
                assert rate == 16000 and joined.digest(dry) == prior["pcm_sha256"]
                assert len(dry) == len(player) and prior["end"] == source["command_at"]
                start = prior["start"]
            noise = np.random.default_rng(18221).standard_normal(len(player))
            mic, ref, clips = handoff.render(dry, player, response, source["echo_gain"], noise)
            assert joined.digest(ref) == result["continue"]["reference_sha256"]
            if role == "fixed":
                assert joined.digest(mic) == result["continue"]["mic_sha256"]
            wav = WORK / f"{len(records):02}.wav"
            sf.write(wav, mic, 16000, subtype="PCM_16")
            files[str(wav)] = joined.sha(wav)
            records.append(dict(voice=source["voice"], carrier=source["carrier"], room=source["room"],
                                start=start, end=source["command_at"], target_id=role, path=str(wav),
                                sha256=files[str(wav)], samples=len(mic), pcm_sha256=joined.digest(mic),
                                before_echo_sha256=joined.digest(dry), reference_sha256=joined.digest(ref),
                                clipped_samples=clips))
    assert len(records) == len(rows) // 5 * 4 == 64
    plan = dict(records=records, runtime=old["runtime"], files=files, threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert plan == joined.read(PLAN)
    else:
        joined.save(PLAN, plan)


def run():
    import run_c161_willow_matched_scenes_v1 as official
    official.PLAN = PLAN
    official.OUT = WORK / "runs"
    official.__file__ = str(RUNNER)
    official.willow()


def case():
    sys.path.insert(0, str((ROOT / "qemu_hey_reference_v2").resolve()))
    import run_willow_mac_matched_prime_v1 as official
    official.PLAN = PLAN
    official.__file__ = str(RUNNER)
    official.case()


if __name__ == "__main__":
    {"prepare": prepare, "run": run, "case": case}[sys.argv.pop(1)]()
