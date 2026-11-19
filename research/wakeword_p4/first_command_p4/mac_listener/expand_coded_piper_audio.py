"""Rebuild the emulated four-voice command recordings from two downloadable WAV packs."""
import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from expand_piper_matched_audio import digest, pcm, wav_bytes


def expand(results, output, pair=None):
    stem = "4090-c161-piper-matched-coded"
    report = json.loads((results / (stem + ".json")).read_bytes())
    cue_pack = results / (stem + "-cues.zip")
    assert digest(cue_pack.read_bytes()) == report["cue_archive_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(cue_pack) as cues, ZipFile(results / "4090-c161-fixed-cue-piper-p4-playback.zip") as bases:
        originals = json.loads(bases.read("manifest.json"))["records"]
        if pair is not None and not 0 <= pair < len(originals):
            raise ValueError(f"Pair must be between 0 and {len(originals) - 1}")
        for row in report["records"]:
            if pair is not None and row["pair"] != pair:
                continue
            source = row["source"]
            base = originals[row["pair"]]
            data = bytearray(pcm(bases.read(base["file"])))
            assert digest(data) == base["pcm_sha256"] and source["command_at"] == base["command_at"]
            cue = pcm(cues.read(source["voice"] + ".wav"))
            start, stop = source["cue_at"] * 2, source["command_at"] * 2
            assert len(cue) == stop - start and digest(cue) == row["cue_sha256"]
            data[base["cue_at"] * 2:start] = bytes(start - base["cue_at"] * 2)
            data[start:stop] = cue
            assert digest(data) == row["pcm_sha256"] and digest(data[stop:]) == row["command_sha256"]
            dest = output / f'{row["pair"]:02}-{source["voice"]}.wav'
            wav = wav_bytes(data)
            if dest.exists():
                if dest.read_bytes() != wav:
                    raise FileExistsError(dest)
            else:
                with NamedTemporaryFile(dir=output, delete=False) as out:
                    out.write(wav)
                try:
                    os.link(out.name, dest)
                except FileExistsError:
                    if dest.read_bytes() != wav:
                        raise
                finally:
                    Path(out.name).unlink(missing_ok=True)
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", type=int, help="command index, 0 to 17")
    args = parser.parse_args()
    print("Verified WAVs", expand(args.results, args.output, args.pair))
