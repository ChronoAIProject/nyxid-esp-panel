"""Keep the fixed v5 raw trigger and compare raw/denoised digital command audio at that exact clock."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import gzip

from faster_whisper import WhisperModel
from scipy.signal import fftconvolve
import soundfile as sf

import check_room_rnnoise as rn
import compare_selected_room_transcripts as readers

ch, room, gap = rn.ch, rn.room, rn.gap
OUT = rn.RESULT_DIR / "4090-c161-fixed-v5-rnnoise-uplink.json"
FIRST = 64


def inputs():
    cue, rate = sf.read(gap.device.host.fixed.CUE, dtype="int16")
    assert rate == 16000
    for s, previous, kind, h, bg, cmd in room.room_inputs():
        route = previous["routes"]["fixed_v5"]
        signal = bg + cmd
        at = s["end"] - len(cue)
        wet = fftconvolve(cue.astype(float), h)
        signal[at:at+len(wet)] += wet
        pcm = room.quantize(signal * s["gain"])
        assert ch.digest(pcm) == route["pcm_sha256"] and at == route["cue_at"]
        assert at <= route["opened"] < s["end"] and route["fixed"]
        yield dict(kind=kind, role="fixed_v5", opened=route["opened"], start=at, end=s["end"],
                   command=s["carrier"], voice=s["voice"], pcm_sha256=route["pcm_sha256"],
                   raw_capture=route["capture_id"]), pcm


def main():
    if rn.archived(OUT):
        return
    rn.collect(inputs(), FIRST)
    frozen = ch.read(readers.OUT)
    previous = ch.read(room.parent.OUT)
    assert frozen["sources"][room.parent.OUT.name] == ch.sha(room.parent.OUT)
    reader, small = ch.load_reader()
    assert small == frozen["small_files"]
    info = frozen["turbo"]
    assert all(ch.sha(readers.audit.TURBO/n) == h for n,h in info["files"].items())
    turbo = WhisperModel(str(readers.audit.TURBO), device="cuda", compute_type="float16", num_workers=1, cpu_threads=1)
    decoder = gap.uplink.decoder_api()
    firmware = ch.sha(rn.ELF)
    tally, records = defaultdict(lambda: defaultdict(Counter)), []
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with TemporaryDirectory(dir=rn.WORK) as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for i, (s, pcm) in enumerate(inputs(), FIRST):
            data = gzip.decompress(rn.cache_file(i,s,firmware).read_bytes())
            y, service, heap = rn.parse(data, i, len(pcm))
            logs.writestr(f"uart/{i:02}.log", data)
            cap = gap.replay.capture(reader, decoder, work, logs, i, "fixed-rn", y, s["opened"], s["command"])
            row = previous["records"][i-FIRST]["routes"]["fixed_v5"]
            assert row["opened"] == s["opened"] and row["capture"]["first"] == cap["first"]
            raw = frozen["fixed_v5"][str(s["raw_capture"])]
            assert raw["decoded_sha256"] == row["capture"]["lanes"]["selected"]["decoded_sha256"]
            records.append(dict(source=s, id=i, samples=len(pcm), output_pcm_sha256=ch.digest(y),
                                service=rn.service_stats(service), heap=heap, capture=cap, raw=raw))
        for p in sorted(work.glob("*-attempt*.gz")):
            logs.write(p, "opus-retry/"+p.name)
        for p in sorted(rn.cache_file(FIRST,s,firmware).parent.glob("*.attempt-*.gz")):
            if int(p.name.split("-")[0]) >= FIRST:
                logs.write(p, "frontend-retry/"+p.name)
    with ZipFile(tmp) as logs:
        assert len(records) == 16 and logs.testzip() is None
        for r in records:
            i, cap, s = r["id"], r["capture"], r["source"]
            data = logs.read(f"opus/{i:03}-fixed-rn.log").decode().replace("\r\n", "\n")
            for lane, label in enumerate(("selected","raw")):
                packets = gap.uplink.packets(data, i, cap["emitted"]-cap["first"], lane)
                wave = gap.uplink.decode(decoder, packets)
                saved = cap["lanes"][label]
                assert ch.digest(wave) == saved["decoded_sha256"]
                if label == "selected":
                    r["rnnoise"] = dict(small=saved["asr"], turbo=gap.uplink.recognition(turbo,wave,s["command"]))
            for route in ("raw", "rnnoise"):
                for size in ("small","turbo"):
                    tally[s["kind"]][route][size] += int(r[route][size]["complete_suffix"])
            filtered, _, _ = rn.parse(logs.read(f"uart/{i:02}.log"), i, r["samples"])
            assert ch.digest(filtered) == r["output_pcm_sha256"]
    tmp.replace(archive)
    ch.save(OUT, dict(records=records, summary=tally, asr=small, turbo=info, elf_sha256=firmware,
        firmware=gap.uplink.firmware(), archive_sha256=ch.sha(archive), sources={p.name:ch.sha(p) for p in
        (Path(__file__),Path(rn.__file__),room.parent.OUT,readers.OUT,gap.device.host.fixed.CUE)},
        policy="Raw fixed matcher decision from the identical room waveform. Keep start-of-command and two-second "
        "lookback fixed across lanes. RNNoise supplies only the P4 Opus audio, with its full causal delay. "
        "Timing and buffer interaction of simultaneous matcher, encoder and denoiser require hardware."))


if __name__ == "__main__":
    main()
