"""Replay the original three unmarked 16 kHz files from the Mac rev1 board trials."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf
import score_nicole_binary40 as analysis
import fixed_rearm_demand_p4 as p4

WORK = Path(".local/wakeword/mac-unmarked")
INPUT = WORK / "mac.zip"
URL = ("https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/"
       "rev1-unmarked-marin-echo-sage-mac-replay-v1.zip")
SOURCE = analysis.RESULTS / "4090-mac-unmarked-inputs.json"
OUT = analysis.RESULTS / "4090-mac-unmarked-binary40.json"
DIGESTS = {
    "marin": "ea3b02ce55e0ec88241fa18fc031d6dffba176cac42fdfa1e2798aacc3389e3f",
    "echo": "3771b79378b24cafa58ff60ccbf9b1c1c4d41becac382f959d3b3ca7af641d3c",
    "sage": "284ee0642cb8fce7efa2557d87da2eb21f0591acbb137120468018235002aa10",
}


def main():
    archive_hash = analysis.P.sha(INPUT)
    assert archive_hash == "f24bb596dd4d5de56272718e7d7c531127cd2359b516c14ac98d468558579eae"
    build = analysis.P.read(p4.WORK / "build.json")
    assert build == analysis.P.read(analysis.RESULTS / "4090-nicole-context.json")["p4"]
    rows, hashes = [], {}
    trace = SOURCE.with_suffix(".zip")
    with ZipFile(INPUT) as z, ZipFile(trace, "w", ZIP_DEFLATED) as uart:
        manifest = json.loads(z.read("manifest.json"))
        recordings = manifest["synthetic_development_recordings"]
        assert [r["voice"] for r in recordings] == list(DIGESTS)
        for record in recordings:
            voice = record["voice"]
            filename = voice + "-hey-nyx.wav"
            assert record["path"] == filename
            data = z.read(filename)
            digest = sha256(data).hexdigest()
            assert digest == DIGESTS[voice] == record["sha256"]
            with sf.SoundFile(BytesIO(data)) as f:
                assert (f.samplerate, f.channels, f.subtype) == (16000, 1, "PCM_16")
                pcm = f.read(dtype="int16")
            assert len(pcm) == record["samples"]
            path = WORK / filename
            path.write_bytes(data)
            row = dict(label=path.stem, path=str(path), samples=len(pcm), probe_start=0,
                       sha256=digest, pcm_sha256=analysis.P.digest(pcm))
            for name in ("fixed", "c161"):
                result, log = p4.replay(name, row, WORK/name/voice)
                row[name] = result
                member = f"{name}/{voice}.log"
                hashes[member] = analysis.P.sha(log)
                uart.write(log, member)
            rows.append(row)
    analysis.P.save(SOURCE, dict(records=rows, source_url=URL, archive_sha256=archive_hash, p4=build,
                                 negative_playlist=manifest["daily_negative_playlist"], uart=trace.name,
                                 uart_sha256=analysis.P.sha(trace), files=hashes))
    analysis.run(SOURCE, INPUT, OUT, WORK, "please repeat exactly purple elephant", {r["label"] for r in rows})


if __name__ == "__main__":
    main()
