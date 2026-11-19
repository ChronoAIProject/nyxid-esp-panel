"""Match received wake-to-interference ratio to the handoff cue in each panel speech scene."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import handoff_willow as original

PARENT = original.PLAN
BASE = original.BASE.with_name(original.BASE.name + "-level")
PLAN = BASE.with_name(BASE.name + "-plan.json")
WORK = Path(".local/wakeword/c161-handoff-willow-level")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import coded_close_join as joined

    prior = joined.read(PARENT)
    files = {p: prior["files"][p] for p in prior["runtime"].values()}
    for path in (RUNNER, PARENT, Path(original.__file__), Path(joined.matched.official.official.__file__),
                 joined.matched.official.official.SOURCE):
        files[str(path)] = joined.sha(path)
    rows = []
    WORK.mkdir(exist_ok=True)
    for i in range(0, len(prior["records"]), 4):
        scene = prior["records"][i:i+4]
        assert [r["target_id"] for r in scene] == ["fixed", "ordinary", "willow", "silence"]
        waves = {}
        for r in scene:
            samples, rate = sf.read(r["path"], dtype="int16")
            assert rate == 16000 and joined.digest(samples) == r["pcm_sha256"] and r["clipped_samples"] == 0
            waves[r["target_id"]] = samples.astype(float)
        background = waves["silence"]
        for r in scene[:3]:
            role = r["target_id"]
            cue = waves[role] - background
            assert not cue[:r["start"]].any()
            window = slice(r["start"], r["end"])
            interference = np.linalg.norm(background[window])
            ratio = np.linalg.norm(cue[window]) / interference
            if role == "fixed":
                reference_ratio = ratio
                continue
            gain = reference_ratio / ratio
            mix = np.rint(background + gain*cue)
            clips = int(np.count_nonzero((mix < -32768) | (mix > 32767)))
            mic = mix.clip(-32768, 32767).astype("<i2")
            measured = np.linalg.norm((mic - background)[window]) / interference
            path = WORK / f"{len(rows):02}.wav"
            sf.write(path, mic, 16000, subtype="PCM_16")
            files[str(path)] = joined.sha(path)
            rows.append(dict(r, original_mic_sha256=r["pcm_sha256"], silence_mic_sha256=scene[3]["pcm_sha256"],
                        fixed_mic_sha256=scene[0]["pcm_sha256"], cue_gain_db=float(20*np.log10(gain)),
                        original_sir_db=float(20*np.log10(ratio)), target_sir_db=float(20*np.log10(reference_ratio)),
                        measured_sir_db=float(20*np.log10(measured)), clipped_samples=clips,
                        path=str(path), sha256=files[str(path)], pcm_sha256=joined.digest(mic)))
    assert len(rows) == 32
    result = dict(runtime=prior["runtime"], threshold_requests=["DEFAULT"], files=files, records=rows,
        policy="Post-hoc received-level diagnostic. Subtract the paired silent-cue mic to isolate the wake "
               "and its decay. Match its RMS ratio to the fixed cue over each role's own scheduled wake interval. "
               "Retain the same command, ambient noise, panel speech and electrical reference.")
    if PLAN.exists():
        assert joined.read(PLAN) == result
    else:
        joined.save(PLAN, result)


def score():
    import numpy as np
    import soundfile as sf
    import coded_close_join as joined
    import handoff_barge_in as handoff
    from score_handoff_willow import willow_events

    plan = joined.read(PLAN)
    plan_digest = joined.sha(PLAN)
    parent = joined.read(original.BASE.with_suffix(".json"))
    old = {(r["room"], r["voice"], r["carrier"], r["role"]): r for r in parent["records"]}
    rows = []
    archive = BASE.with_suffix(".zip")
    partial = BASE.with_suffix(".tmp")
    with TemporaryDirectory() as tmp, ZipFile(partial, "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        detector = joined.binding(work/"full.so", coded="v5")
        pair = joined.binding(work/"pair.so", coded="aligned")
        for source, _, player, _ in handoff.scenes():
            noise = np.random.default_rng(18221).standard_normal(len(player))
            ref = handoff.bench.channel(player, "noisy_line", noise)
            for role in ("ordinary", "willow"):
                i = len(rows)
                r = plan["records"][i]
                assert r["target_id"] == role
                key = tuple(r[k] for k in ("room", "voice", "carrier"))
                assert key == tuple(source[k] for k in ("room", "voice", "carrier"))
                before = old[(*key, role)]
                assert before["mic_sha256"] == r["original_mic_sha256"]
                mic, rate = sf.read(r["path"], dtype="int16")
                assert rate == 16000 and joined.digest(mic) == r["pcm_sha256"]
                assert joined.digest(ref) == r["reference_sha256"]
                events = willow_events(i, r, plan_digest, logs, WORK/"runs")
                nyx = handoff.probe.score(work, source, mic, ref, detector, i, build=handoff.test.BUILD,
                        prefix=pair, qualify=handoff.probe.qualify, wait_for_third=True,
                        frame_activity=True, qualify_capture_frame=True)
                logs.write(work/"serial.log", f"{i:02}/nyx-serial.log")
                rows.append(dict(input=r, willow=events, c161=nyx["c161"], fixed=nyx["fixed"],
                                 opened=nyx["opened"], nyx_uart_sha256=nyx["uart_sha256"]))
                print(i, role, events, nyx["opened"], flush=True)
    assert len(rows) == len(plan["records"])
    partial.replace(archive)
    joined.save(BASE.with_suffix(".json"), dict(records=rows, plan_sha256=plan_digest, archive_sha256=joined.sha(archive),
        parent_sha256=joined.sha(original.BASE.with_suffix(".json")), firmware_sha256=joined.sha(handoff.test.BUILD/"nyx_mac_listener.bin")))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action in ("case", "run"):
        original.PLAN, original.WORK, original.RUNNER = PLAN, WORK, RUNNER
        {"case": original.case, "run": original.run}[action]()
    else:
        {"prepare": prepare, "score": score}[action]()
