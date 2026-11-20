"""Move the same added waveform to three shorter advances and check panel and player paths."""
from collections import Counter, defaultdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import bella_early_marker as early

P, native = early.P, early.native
BASE = early.BASE.with_name("4090-bella-short-markers")
PLAN = BASE.with_name(BASE.name + "-plan.json")
OUT = BASE.with_suffix(".json")
WORK = Path(".local/wakeword/bella-short-markers")
ADVANCES = (100, 150, 200)


def cue_path(ms):
    return BASE.with_name(f"hye-nyx-af-bella-{ms}ms-experiment.wav")


def prepare():
    old, rate = sf.read(early.WAV, dtype="int16")
    assert rate == 16000 and P.sha(early.WAV) == P.read(early.FIT)["cue_sha256"]
    original, base, db = early.playback.candidate()
    assert db == 5 and P.digest(base) == P.read(early.playback.OUT)["pcm_sha256"]
    source = old.astype(float) - base
    WORK.mkdir(parents=True, exist_ok=True)
    staging = early.fitted
    items, inputs = [], []
    for ms in ADVANCES:
        offset = (400-ms)*16
        addition = np.zeros_like(source)
        assert not np.any(source[-offset:])
        addition[offset:] = source[:-offset]
        up, down = addition > 0, addition < 0
        scale = min(1., ((32767.-base[up])/addition[up]).min(), ((-32768.-base[down])/addition[down]).min())
        shaped = np.rint(base.astype(float) + scale*addition)
        assert 0 < scale <= 1 and shaped.min() >= -32768 and shaped.max() <= 32767
        cue = shaped.astype("<i2")
        wav = cue_path(ms)
        wav.write_bytes(early.playback.playback.sources.wav(cue, "PCM_16"))
        work = WORK / str(ms)
        work.mkdir(exist_ok=True)
        items.append(dict(advance_ms=ms, addition_scale=scale, wav=str(wav), wav_sha256=P.sha(wav),
                          pcm_sha256=P.digest(cue), peak=int(np.abs(shaped).max()), samples=len(cue)))
        staging.WORK, staging.PLAN, staging.FIT, staging.WAV = work, work/"weak.json", work/"fit.json", wav
        P.save(staging.FIT, dict(cue_sha256=P.sha(wav)))
        staging.prepare()
        for r in P.read(staging.PLAN)["records"]:
            inputs.append(dict(r, label=f"{ms}-"+r["label"], group="external", advance_ms=ms))
        for key, r, mic, ref in early.playback.playback.panel(original, cue):
            label = f"{ms}-"+key.replace("/", "-")
            mic_path, ref_path = work/(label+".wav"), work/(label+"-ref.wav")
            sf.write(mic_path, mic, 16000, subtype="PCM_16")
            sf.write(ref_path, ref, 16000, subtype="PCM_16")
            inputs.append(dict(r, advance_ms=ms, label=label, path=str(mic_path), reference_path=str(ref_path),
                               pcm_sha256=P.digest(mic), ref_sha256=P.digest(ref), sha256=P.sha(mic_path)))
    assert len(inputs)==len({r["label"] for r in inputs})==69
    files = (early.FIT, early.WAV, early.playback.OUT, Path(__file__), Path(staging.__file__),
             early.ROOT/"experiments/bella_short_marker_protocol.md")
    P.save(PLAN, dict(records=inputs, candidates=items, sources={str(p):P.sha(p) for p in files}))


def report():
    from zipfile import ZipFile
    plan, decision = P.read(PLAN), P.read(WORK/"dual.json")
    assert P.sha(PLAN)==decision["plan_sha256"] and P.sha(WORK/"dual.zip")==decision["archive_sha256"]
    counts, rows, files = defaultdict(Counter), [], {}
    with ZipFile(WORK/"dual.zip") as logs:
        for r, result in zip(plan["records"], decision["records"], strict=True):
            assert r["label"]==result["label"]
            c = counts[str(r["advance_ms"])]
            ev = result["events"]
            if r["group"] == "external":
                timed = early.playback.selection.first_read(ev["O"], r)
                c[r["rate"]] += int(timed["timely"])
            else:
                timed = None
                c[r["role"]] += int(bool(ev["O"]))
            rows.append(dict(**r, native=result, read=timed))
            for name, sha in result["members"].items():
                assert early.playback.wakes.digest(logs.read(name)) == sha
                files[name] = sha
    qualifies = [ms for ms in ADVANCES if counts[str(ms)]["synchronous"]==0 and counts[str(ms)]["silent"]==4
                 and all(counts[str(ms)][tag]==3 for tag,_ in early.fitted.gain_grid.SPEEDS)]
    OUT.with_suffix(".zip").write_bytes((WORK/"dual.zip").read_bytes())
    P.save(OUT,dict(records=rows, counts=counts, selected_ms=min(qualifies,default=None), candidates=plan["candidates"],
                    files=files, archive_sha256=decision["archive_sha256"], plan_sha256=P.sha(PLAN),
                    model_sha256=decision["model_sha256"], guest_sha256=decision["guest_sha256"], qemu_sha256=decision["qemu_sha256"]))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "dual":
        native.PLAN, native.WORK = PLAN, WORK
        native.dual()
    else:
        {"prepare": prepare, "report": report}[action]()
