"""Compare alternative echo-cancelled uplinks at the same measured fixed-cue OPEN while Bot speech continues."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from faster_whisper import WhisperModel
import numpy as np
import torch

import check_fixed_cue_playback_stop as stop
import replay_c161_speex_p4 as speex
from score_c161_lexical_echo import subtract_playback

fixed, spoken, ch, uplink = stop.fixed, stop.spoken, stop.ch, stop.uplink
OUT = ch.RESULTS / "4090-c161-fixed-cue-echo-uplinks.json"


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    fw = uplink.firmware()
    parent = ch.read(fixed.OUT)
    previous = {r["mic_sha256"]: r for r in parent["records"] if r["source"]["mode"] == "overlap"}
    assert all(parent[k] == v for k, v in fw.items())
    build = speex.OUT / "build-float"
    speex_report = ch.RESULTS / (speex.STEM + "-float.json")
    identity = ch.read(speex_report)
    assert ch.sha(build / "nyx_speex_echo.elf") == identity["elf_sha256"]
    assert ch.sha(build / "nyx_speex_echo.bin") == identity["app_sha256"]
    ordinary, asr = ch.load_reader()
    assert parent["asr"] == identity["asr"] == asr
    files = ch.read(ch.RESULTS / (stop.audit.STEM + ".json"))["turbo"]["files"]
    assert all(ch.sha(stop.audit.TURBO / k) == v for k, v in files.items())
    turbo = WhisperModel(str(stop.audit.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = uplink.decoder_api()
    archive = OUT.with_suffix(".zip")
    records = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        for meta, mic, ref in spoken.extra_cases():
            if meta.get("role") != "fixed-v5" or meta["mode"] != "overlap":
                continue
            old = previous[ch.digest(mic)]
            assert old["source"] == meta and ch.digest(ref) == old["ref_sha256"]
            index = old["index"]
            uart = work / "speex.log"
            payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
            uplink.boot(build, uart, b"SPEEX PASS", payload, work)
            cleaned, costs, heap = speex.parse(uart, len(mic))
            residual, clips = subtract_playback(mic, ref)
            first, emitted = old["first"], old["emitted"]
            captures = (cleaned[first:emitted], residual[first:emitted])
            assert all(len(c) == old["count"] for c in captures)
            codec = work / "opus.log"
            payload = struct.pack("<3I", 0x3552504f, index, old["count"]) + b"".join(c.tobytes() for c in captures)
            uplink.boot(uplink.CODEC, codec, b"UPLINK PASS", payload, work)
            logs.write(codec, f"{index:02}-opus.log")
            text = codec.read_text()
            lanes = {}
            for lane, (name, capture) in enumerate(zip(("speex_p4", "fir_gpu"), captures, strict=True)):
                packets = uplink.packets(text, index, old["count"], lane)
                audio = uplink.decode(decoder, packets)
                views = {}
                for key, offset in (("full", 0), ("from_open_minus_100ms", old["open"][0] - first - 1600)):
                    views[key] = {}
                    for model, reader in (("small.en", ordinary), ("large-v3-turbo", turbo)):
                        reading = uplink.recognition(reader, audio[offset:], meta["carrier"])
                        views[key][model] = {k: reading[k] for k in ("text", "complete_suffix", "exact")}
                lanes[name] = dict(input_sha256=ch.digest(capture), decoded_sha256=ch.digest(audio),
                                   packets=len(packets), views=views)
            records.append(dict(index=index, source=meta, mic_sha256=ch.digest(mic), ref_sha256=ch.digest(ref),
                                open=old["open"], first=first, emitted=emitted, count=old["count"], lanes=lanes,
                                speex_pcm_sha256=ch.digest(cleaned), speex_uart_sha256=ch.sha(uart),
                                speex_heap=heap, speex_frame_us=dict(median=float(np.median(costs)),
                                p95=float(np.percentile(costs,95)), maximum=max(costs)), fir_clips=clips))
            print(index, flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for failure in failures:
            logs.write(failure, failure.name)
    assert len(records) == 12
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for row in records:
            i = row["index"]
            data = logs.read(f"{i:02}-opus.log").decode().replace("\r\n", "\n")
            for lane, name in enumerate(("speex_p4", "fir_gpu")):
                packets = uplink.packets(data, i, row["count"], lane)
                assert ch.digest(uplink.decode(decoder, packets)) == row["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=records, **fw, speex_elf_sha256=identity["elf_sha256"], asr=asr, turbo_files=files,
                      sources={p.name:ch.sha(p) for p in (Path(__file__), fixed.OUT, speex_report)},
                      archive_sha256=ch.sha(archive), retries=[p.name for p in failures],
                      policy="Preserve full ongoing Bot playback. Keep the actual signed-reference P4 listener "
                      "OPEN and two-second capture bounds. Replace only the uplink audio, using a separate P4 "
                      "SpeexDSP float firmware or the existing CUDA causal FIR. Encode each in P4 Opus, decode "
                      "twice, and transcribe with two unprompted GPU readers. Alternative DSP and listener "
                      "were run separately; combined whole-board resources and scheduling require hardware."))


if __name__ == "__main__":
    main()
