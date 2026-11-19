"""Build the fixed af_bella speaker tracks from the measured dry PCM and original command clocks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RESULTS = Path(__file__).resolve().parents[2] / "results"
REPORT = RESULTS / "4090-c161-bella-command-prefix.json"
TRACKS = RESULTS / "c161-eight-voice-marker-gains.zip"
CUE = RESULTS / "hye-nyx-af-bella-marker-rms-v1.wav"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pcm(data):
    with wave.open(io.BytesIO(data)) as src:
        assert (src.getnchannels(), src.getsampwidth(), src.getframerate(), src.getcomptype()) == (1, 2, 16000, "NONE")
        frames = src.readframes(src.getnframes())
        assert len(frames) == 2 * src.getnframes()
        return frames


def put(archive, name, data):
    archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)


def build(output):
    report = json.loads(REPORT.read_bytes())
    plan = {(r["batch"], r["source"]["index"]): r for r in report["records"] if r["paired_back_grid"]}
    assert len(plan) == 24 and digest(TRACKS.read_bytes()) == report["sources"][TRACKS.name]
    cue_data = CUE.read_bytes()
    assert digest(cue_data) == report["sources"][CUE.name]
    cue = pcm(cue_data)
    manifest = dict(format="16 kHz mono PCM16", clocks="zero-based samples in each dry playback file",
                    cue=CUE.name, cue_sha256=digest(cue_data), original_pack_sha256=report["sources"][TRACKS.name],
                    report_sha256=digest(REPORT.read_bytes()), records=[])
    with TemporaryDirectory(dir=output.parent) as folder, ZipFile(TRACKS) as previous:
        partial = Path(folder) / output.name
        with ZipFile(partial, "w") as archive:
            for row in json.loads(previous.read("manifest.json"))["records"]:
                case = plan.pop((row["batch"], row["index"]))
                s = case["source"]
                entry = row["connected"]
                data = previous.read(entry["file"])
                assert digest(data) == entry["sha256"]
                pivot = row["command_at"]
                assert pivot == s["command_at"] and pivot * 2 >= len(cue)
                suffix = pcm(data)[2*pivot:]
                assert digest(suffix) == s["command_pcm_sha256"]
                output_pcm = bytes(2*pivot-len(cue)) + cue + suffix
                assert digest(output_pcm) == s["dry_pcm_sha256"]
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(output_pcm)
                name = f'{row["batch"]}/{row["index"]:03}.wav'
                put(archive, name, buf.getvalue())
                manifest["records"].append(dict(file=name, sha256=digest(buf.getvalue()), pcm_sha256=digest(output_pcm),
                    cue_file_start=s["cue_file_start"], command_at=pivot, command=row["command"], command_voice=row["voice"],
                    command_pcm_sha256=digest(suffix), paired_original=entry["file"], paired_original_sha256=entry["sha256"],
                    simulated_external_db=s["external_db"], simulated_tx_sample=s["transmit_event"]))
            assert not plan
            put(archive, "manifest.json", (json.dumps(manifest, indent=2)+"\n").encode())
        with ZipFile(partial) as archive:
            assert archive.testzip() is None and len(archive.namelist()) == 25
        partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
