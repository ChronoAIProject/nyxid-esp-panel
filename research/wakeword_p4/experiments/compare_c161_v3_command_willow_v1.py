"""Compare the frozen playback cue and same-voice Willow on the immediate-command trials."""
import hashlib
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

import compare_c161_cue_willow_v1 as official
import replay_c161_pcm_cues_v1 as matcher
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-v3-command-willow"
PLAN = RESULTS / f"{STEM}-plan-v1.json"
OUT = Path(".local/wakeword/c161-v3-command-willow-v1")


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def make():
    previous = RESULTS / "4090-c161-v3-command-playback-v1.json"
    paired = read(previous)
    original = {(r["voice"], r["scene"], r["channel"], r["playing"]): r for r in paired["records"]}
    sources = RESULTS / "4090-c161-signed-forward-plan-v1.json"
    assert sha(sources) == paired["source_plan_sha256"]
    plan = read(sources)
    inputs = RESULTS / "4090-c161-signed-forward-inputs-v1.zip"
    assert sha(inputs) == plan["archive_sha256"] == paired["source_archive_sha256"]
    old = RESULTS / "4090-c161-cue-willow-plan-v1.json"
    dry = read(old)["records"][1]
    assert dry["target_id"] == "willow" and not dry["echo"] and dry["noise_nominal"] is None
    willow = load_wav(dry)[dry["start"]:dry["end"]]
    cue = load_wav(dict(path=RESULTS / "hye-nyx-c161-synthetic-cue-v3.wav", sha256=paired["cue_sha256"]))
    assert (len(cue), len(willow)) == (21200, 22400)
    rows = []
    files = {str(p): sha(p) for p in [previous, sources, inputs, old, Path(__file__).resolve(),
             Path(official.runner.__file__).resolve(), Path(official.runner.official.__file__).resolve(),
             official.runner.official.SOURCE]}
    for f in plan["runtime"].values():
        assert sha(Path(f)) == plan["files"][f]
        files[f] = plan["files"][f]
    OUT.mkdir(exist_ok=True)
    with ZipFile(inputs) as archive:
        for idx, r in enumerate(plan["records"]):
            if r["target_id"] != "nyx":
                continue
            raw = archive.read(Path(r["path"]).name)
            assert hashlib.sha256(raw).hexdigest() == r["sha256"]
            pcm, rate = sf.read(io.BytesIO(raw), dtype="int16")
            assert rate == 16000 and digest(pcm) == r["pcm_sha256"]
            start = r["start"]
            command = pcm[r["first_phone"]:]
            count = start + len(command)
            noise = np.random.default_rng(85370+idx).normal(size=count+len(willow))
            noise = lfilter([1.], [1., -.75], noise)
            noise *= np.sqrt(np.mean(cue.astype(float)**2)/np.mean(noise[:count+len(cue)]**2)) * .065
            for channel in ("digital", "room_noise"):
                prefix = None
                for role, target in (("nyx", cue), ("willow", willow), ("silence", np.zeros_like(cue))):
                    full = np.r_[pcm[:start], target, command]
                    if channel == "room_noise":
                        ir = np.zeros(577)
                        ir[[0,173,576]] = [.65,.24,-.13]
                        full = lfilter([.25,.5,.25],[1.],lfilter(ir,[1.],full)) + noise[:len(full)]
                    clipped = int(np.count_nonzero((full > 32767) | (full < -32768)))
                    full = np.rint(full).clip(-32768,32767).astype("<i2")
                    if prefix is None:
                        prefix = full[:start]
                    np.testing.assert_array_equal(prefix, full[:start])
                    if role != "willow":
                        ref = original[(r["voice"], r["scene"], channel, role == "nyx")]
                        assert digest(full) == ref["pcm_sha256"]
                    path = OUT / f"{len(rows):02d}.wav"
                    sf.write(path, full, 16000, subtype="PCM_16")
                    files[str(path)] = sha(path)
                    rows.append(dict(path=str(path), sha256=files[str(path)], samples=len(full), pcm_sha256=digest(full),
                                     target_id=role, prefix_sha256=digest(prefix), carrier=r["carrier"], voice=r["voice"],
                                     scene=r["scene"], channel=channel, start=start, end=start+len(target), clipped=clipped))
    assert len(rows) == 72
    save(PLAN, dict(records=rows, files=files, runtime=plan["runtime"], threshold_requests=["DEFAULT"],
                    nyx_sha256=digest(cue), willow_sha256=digest(willow)))


def review():
    ref = RESULTS / f"{STEM}-v1.json"
    before = RESULTS / "4090-c161-v3-command-playback-v1.json"
    old = {r["pcm_sha256"]: r for r in read(before)["records"]}
    rows = []
    with tempfile.TemporaryDirectory() as work, patch.object(matcher, "LIB", Path(work)/"confirmed.so"):
        det = matcher.binding(True)
        for r in read(ref)["records"]:
            confirmed = matcher.detect(det, (load_wav(r),))
            hits = matcher.joined(r["original_events"], confirmed)
            if r["target_id"] != "willow":
                prev = old[r["pcm_sha256"]]
                assert (confirmed, hits, r["original_events"]) == (prev["confirmed"], prev["merged"], prev["original"])
            rows.append(dict(voice=r["voice"], scene=r["scene"], channel=r["channel"], target=r["target_id"],
                             start=r["start"], end=r["end"], pcm_sha256=r["pcm_sha256"], confirmed=confirmed,
                             nyx=hits, willow=r["official_events"]))
    save(RESULTS / f"{STEM}-review-v1.json", dict(official_report_sha256=sha(ref), playback_sha256=sha(before), records=rows))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    with patch.multiple(official, PLAN=PLAN, STEM=STEM, CASES=OUT/"official", __file__=__file__):
        {"make":make, "case":official.case, "run":official.execute, "summarize":official.summarize,
         "review":review}[action]()
