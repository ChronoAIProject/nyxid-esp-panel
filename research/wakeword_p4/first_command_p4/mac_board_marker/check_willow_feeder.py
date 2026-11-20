"""Replay the twelve archived official Willow call sequences through the actual panel feeder."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import wave

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "results/4090-willow-mac-matched-prime-evidence-v1.tar.gz"


def replay(board, fixtures, work):
    source = (board / "components/panel_services/panel_voice.c").read_text()
    first = source.index("enum { WILLOW_PRIME_SAMPLES")
    first = source.rindex("#ifdef NYX_PANEL_RESEARCH_WILLOW", 0, first)
    end = source.index("#ifdef NYX_PANEL_RESEARCH_WAKE_TIMING", first)
    begin = source.index("static inline int feed_wake_model", end)
    stop = source.index("#ifdef NYX_PANEL_RESEARCH_FIXED_MARKER", begin)
    (work / "willow_functions.inc").write_text(source[first:end] + source[begin:stop])
    exe = work / "feeder"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Werror", "-O2", "-DNYX_PANEL_RESEARCH_WILLOW=1",
                    "-I", work, HERE / "willow_feeder.c", "-o", exe], check=True)
    rows = []
    with tarfile.open(ARCHIVE) as archive:
        for index in range(12):
            report = json.load(archive.extractfile(f"{index:03}/report.json"))
            returns = archive.extractfile(f"{index:03}/raw-returns.bin").read()
            assert report["raw_returns_sha256"] == sha256(returns).hexdigest()
            prime = bytes(result+1 for _, result in report["prime_returns"])
            row = report["record"]
            with wave.open(str(fixtures / Path(row["path"]).name)) as wav:
                assert wav.getparams()[:3] == (1, 2, 16000)
                raw = wav.readframes(wav.getnframes())
            assert sha256(raw).hexdigest() == report["source_pcm_sha256"]
            samples = len(raw)//2 + report["final_padding_samples"]
            pcm = raw + bytes(2*report["final_padding_samples"])
            all_returns = prime + returns
            assert len(all_returns)*512 == 81600 + samples
            native = [e["relative_sample"] for e in report["real_events"] if e["result"] == 1]
            verified = all_returns.count(0)
            for n in (256, 1024):
                out = subprocess.check_output([exe, str(samples), str(n)], input=pcm+all_returns)
                events = [tuple(map(int, line.split())) for line in out.splitlines()]
                assert events == [(hit, min(samples, (hit + n-1)//n*n)) for hit in native]
            rows.append(dict(index=index, target=row["target"], voice=row["voice"], form=row["form"],
                             pcm_sha256=report["source_pcm_sha256"], calls=len(all_returns),
                             channel_verified_returns=verified, model_ends=native,
                             read_ends_256=[(hit+255)//256*256 for hit in native]))
    # Exercise valid -1 both at warmup and on live samples independently of fixture coverage.
    samples = 2048 - 192
    returns = bytearray([1]*163)
    returns[0], returns[159], returns[161] = 0, 0, 2
    out = subprocess.check_output([exe, str(samples), "1024"], input=bytes(2*samples)+returns)
    assert out == b"1344 1856\n"
    return dict(source="ESP32 Xtensa DET_MODE_95 DEFAULT archived returns, native host-compiled panel feeder",
                archive_sha256=sha256(ARCHIVE.read_bytes()).hexdigest(), read_sizes=[256, 1024],
                synthetic_channel_verified_check=True, records=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as tmp:
        print(json.dumps(replay(args.board, args.fixtures, Path(tmp)), indent=2))
