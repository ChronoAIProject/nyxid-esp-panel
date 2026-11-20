"""Pack the actual external speaker for the three rate stress cases with paired controls."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import bella_early_marker as early

P, study = early.P, early.fitted.gain_grid
PACK = early.WORK / "hye-nyx-paired-external-player-rates-v1.zip"
OUT = early.BASE.with_name("4090-bella-early-marker-playback-download.json")
RELEASE = "https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/"


def main():
    gain = early.playback.playback.gain
    daily = early.fitted.stems.overlap
    source = P.read(gain.PLAN)
    assert P.sha(gain.PACK) == source["archive_sha256"]
    base = {r["label"]: r for r in P.read(study.PLAN)["records"]}
    erased = {r["label"]: r for r in P.read(study.rates.PLAN)["records"]}
    fitted = {r["label"]: r for r in P.read(early.PLAN)["records"]}
    candidate, sr = sf.read(early.WAV, dtype="int16")
    assert sr == 16000 and P.sha(early.WAV) == P.read(early.PLAN)["cue_sha256"]
    rooms, taps = daily.rooms()
    roomnames = {r["index"]: r["file"] for r in P.read(daily.OUT)["records"] if r["role"]=="original"}
    renderers = {case["file"]: render for case, _, render, _, _, _ in rooms[-6]}
    records = []
    with TemporaryDirectory(dir=early.WORK) as folder, ZipFile(gain.PACK) as inputs:
        temp = Path(folder)
        with ZipFile(temp / PACK.name, "w", ZIP_DEFLATED) as dest:
            for r in source["records"]:
                if r["level"] not in ("erased", "plus5dB"):
                    continue
                scene = r["scene"]
                raw = inputs.read(r["member"])
                assert early.playback.wakes.digest(raw) == r["speaker_sha256"]
                x, sr = sf.read(BytesIO(raw), dtype="int32")
                assert sr == 16000 and np.all(x % 16384 == 0)
                dry = x // 16384
                versions = [(r["role"], dry)]
                if r["role"] == "absent":
                    extra = dry.copy()
                    extra[r["end"]-len(candidate):r["end"]] += candidate.astype(np.int32)
                    versions.append(("early-nyx", extra))
                for role, track in versions:
                    for tag, speed in study.SPEEDS:
                        if role == "absent":
                            target = erased[f"{scene}-erased-{tag}"]
                        elif role == "early-nyx":
                            target = fitted[f"{scene:03}-multirate-{tag}"]
                        else:
                            target = base[f"{scene:03}-{role}-plus5dB-{tag}"]
                        played = track if speed == 1 else np.rint(resample_poly(track.astype(float), speed.denominator, speed.numerator))
                        assert np.abs(played).max() < 131072
                        raw = early.playback.playback.sources.wav(played.astype(np.int32)*16384, "PCM_24")
                        decoded, sr = sf.read(BytesIO(raw), dtype="int32")
                        assert sr == 16000 and np.array_equal(decoded // 16384, played)
                        rear = daily.board.bella.rear_filter(decoded // 16384, taps)
                        rear = np.pad(rear, (0, max(0, len(dry)-len(rear))))[:len(dry)]
                        mic, _ = renderers[roomnames[scene]](rear)
                        assert P.digest(mic) == target["pcm_sha256"] and r["reference_sha256"] == target["ref_sha256"]
                        name = f"{scene:03}/{tag}/{role}.wav"
                        dest.writestr(name, raw)
                        records.append(dict(path=name, role=role, rate=tag, speed=str(speed), scene=scene,
                                       samples=len(played), command=r["command"], command_at=target["end"],
                                       mic_sha256=target["pcm_sha256"], ref_sha256=target["ref_sha256"],
                                       bytes=len(raw), sha256=early.playback.wakes.digest(raw)))
            assert len(records) == len({r["path"] for r in records}) == 60
            manifest = dict(records=records, format="mono PCM24 at 16000 Hz, quarter-scale headroom",
                            room_audio=RELEASE+"c161-bella-weak-speaker-gain-room-audio.zip",
                            original_plus5=Path(P.read(early.playback.OUT)["cue"]).name, experimental=early.WAV.name,
                            instructions="Play exactly one external WAV at normal player speed. Each file already implements the selected rate. "
                            "Use the existing matching music and panel tracks, fixed gains and locations and synchronized sample zero. "
                            "nyx and willow are the +5 dB controls. early-nyx adds three fitted marker segments, absent retains the command and competing speech. "
                            "Use an external speaker for the early-nyx experiment. The panel self-opened in two of four simulated paths with synchronous reference.")
            data = (json.dumps(manifest, indent=2)+"\n").encode()
            dest.writestr("manifest.json", data)
        files = {r["path"]:r["sha256"] for r in records}
        files["manifest.json"] = early.playback.wakes.digest(data)
        P.save(temp/OUT.name, dict(url=RELEASE+PACK.name, archive_sha256=P.sha(temp/PACK.name), bytes=(temp/PACK.name).stat().st_size,
               files=files, sources={str(p):P.sha(p) for p in (early.WAV, early.PLAN, study.PLAN, study.rates.PLAN, gain.PLAN, Path(__file__))}))
        (temp/PACK.name).replace(PACK)
        (temp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
