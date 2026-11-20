"""Replay the two complete ESC-50 bell and alarm categories on the published receivers."""
import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
UPSTREAM = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{COMMIT}"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
ROOT = SCRIPT.parents[2]
WORK = Path(".local/wakeword/esc50-bells")
OUT = ROOT / "results/4090-esc50-bells.json"
PLAN = OUT.with_stem(OUT.stem + "-plan")
GROUPS = ("clock_alarm", "church_bells")


def upstream_bytes(path):
    return subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--retry", "3", "--max-time", "20", f"{UPSTREAM}/{path}"])


def prepare(background):
    import scipy
    from scipy.signal import resample_poly

    P = background.P
    WORK.mkdir(parents=True, exist_ok=True)
    index = upstream_bytes("meta/esc50.csv")
    rows = []
    for item in csv.DictReader(index.decode().splitlines()):
        if item["category"] not in GROUPS:
            continue
        name = item["filename"]
        data = upstream_bytes(f"audio/{name}")
        with sf.SoundFile(BytesIO(data)) as wav:
            assert (wav.channels, wav.samplerate, wav.frames, wav.subtype) == (1, 44100, 220500, "PCM_16")
            source = wav.read(dtype="int16")
        signal = resample_poly(source.astype(np.float64), 160, 441)
        rounded = np.rint(signal)
        pcm = np.clip(rounded, -32768, 32767).astype("<i2")
        assert len(pcm) == 80000
        path = WORK / name
        sf.write(path, pcm, 16000, subtype="PCM_16")
        rows.append(dict(label=path.stem, group=item["category"], source=name, fold=int(item["fold"]),
                         freesound_id=item["src_file"], take=item["take"], source_sha256=sha256(data).hexdigest(),
                         path=str(path), sha256=P.sha(path), samples=len(pcm), pcm_sha256=P.digest(pcm),
                         saturated_on_conversion=int(np.count_nonzero(rounded != pcm))))
    assert all(sum(r["group"] == group for r in rows) == 40 for group in GROUPS)
    assert len({r["pcm_sha256"] for r in rows}) == len({r["label"] for r in rows}) == 80
    indoor = background.indoor
    base = P.read(indoor.PLAN)
    assert base["p4"] == P.read(background.p4.WORK / "build.json")
    files = {path: base["files"][path] for path in base["runtime"].values()}
    for path in (SCRIPT, Path(background.__file__), Path(background.p4.__file__), Path(background.wakes.__file__),
                 indoor.PLAN, background.wakes.room.official.SOURCE, Path(background.wakes.room.official.__file__)):
        files[str(path.resolve().relative_to(Path.cwd()))] = P.sha(path)
    files.update({r["path"]: r["sha256"] for r in rows})
    P.save(PLAN, dict(records=rows, p4=base["p4"], runtime=base["runtime"], threshold_requests=["DEFAULT"],
                      files=files, upstream=UPSTREAM, index_sha256=sha256(index).hexdigest(), scipy=scipy.__version__,
                      numpy=np.__version__, conversion="resample_poly 160/441 at original gain, round and saturate PCM16",
                      protocol="All 40 clock_alarm and 40 church_bells clips across five folds. Prime each listener "
                               "for each clip. Zero reference. Count decisions by file and Freesound source. "
                               "Separate reads ending in padding. Selected short events do not measure hourly FAR."))


def main():
    action = sys.argv.pop(1)
    if action == "case":
        sys.path.insert(0, str(ROOT / "qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
        return

    sys.path.insert(0, str(SCRIPT.parents[1] / "mac_listener"))
    sys.path.append(str(ROOT / "experiments"))
    import fixed_rearm_speech_music as background

    # The speech/music driver takes these paths as module settings. Its native
    # replay, per-file clock checks and archive writer are shared by both studies.
    background.PLAN, background.OUT, background.WORK = PLAN, OUT, WORK
    if action == "prepare":
        prepare(background)
    elif action == "willow":
        background.wakes.willow_cases(PLAN, WORK, SCRIPT)
    else:
        {"p4": background.run_p4, "report": background.report}[action]()


if __name__ == "__main__":
    main()
