"""Run the signed wake listener, ESP-SR, SpeexDSP and actual two-second FIFO in one P4 image."""
from pathlib import Path
import re
import struct
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import check_fixed_cue_over_bot as fixed
import compare_fixed_cue_echo_uplinks as other
from transition import samples

ch, uplink = fixed.ch, fixed.uplink
BUILD = uplink.listener.WORK / "build-speex-capture"
OUT = ch.RESULTS / "4090-c161-joint-speex-capture.json"


def main():
    parent = ch.read(fixed.OUT)
    fw = uplink.firmware()
    assert all(parent[k] == v for k, v in fw.items())
    two = ch.read(other.OUT)
    paired = {r["index"]: r for r in two["records"]}
    previous = {(r["mic_sha256"], r["ref_sha256"]): r for r in parent["records"]}
    app = BUILD / "nyx_mac_listener.bin"
    model = uplink.listener.WORK / "model/hey_nyx_model.bin"
    assert ch.sha(model) == uplink.listener.C161 and app.read_bytes().count(model.read_bytes()) == 1
    old_app = uplink.BUILD / app.name
    src = other.speex.ARCHIVE
    assert ch.sha(src) == other.speex.EXPECTED
    with tarfile.open(src) as bundle:
        for name in ("libspeexdsp/mdf.c", "libspeexdsp/fftwrap.c", "libspeexdsp/smallft.c"):
            relative = "speexdsp-1.2.1/" + name
            assert bundle.extractfile(relative).read() == (src.parent / relative).read_bytes()
    reader, identity = ch.load_reader()
    assert identity == parent["asr"]
    decoder = uplink.decoder_api()
    records = []
    seen = set()
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        for meta, mic, ref in fixed.spoken.extra_cases():
            key = (ch.digest(mic), ch.digest(ref))
            if key not in previous or key in seen:
                continue
            seen.add(key)
            old = previous[key]
            index = old["index"]
            log = work / "listener.log"
            packet = struct.pack("<3I", 0x4452594e, index, len(mic)) + mic.tobytes() + ref.tobytes()
            uplink.boot(BUILD, log, b"LISTENER PASS", packet, work)
            uart = log.read_text()
            assert uplink.listener.values(uart, "READY") == [index, len(mic), 256, 81600]
            heap_before, heap_ready = uplink.listener.values(uart, "SPEEX_HEAP")
            emitted, first, count, blocks, total, peak, overruns, heap = uplink.listener.values(uart, "FINISH")
            assert (emitted, first, count) == (old["emitted"], old["first"], old["count"])
            events = [list(map(int, r.split())) for r in re.findall(r"^HITS (.*)$", uart, re.M)]
            allowed = sorted(map(int, re.findall(r"^(?:V5|C161)_ALLOW (\d+)$", uart, re.M)))
            opens = list(map(int, re.findall(r"^OPEN (\d+) ", uart, re.M)))
            assert allowed == old["allowed"] and opens == old["open"]
            assert [r[1] for r in events if r[1] >= 0] == old["c161"]
            assert [r[2] for r in events if r[2] >= 0] == old["v5"]
            tags = ("READY ", "SPEEX_HEAP ", "HITS ", "CUE_ROUTE ", "C161_ALLOW ", "V5_ALLOW ", "OPEN ", "FINISH ", "LISTENER PASS")
            logs.writestr(f"{index:02}-events.txt", "\n".join(r for r in uart.splitlines() if r.startswith(tags)) + "\n")
            row = dict(index=index, mic_sha256=key[0], ref_sha256=key[1], source=meta, open=opens, allowed=allowed,
                       first=first, emitted=emitted, count=count, blocks=blocks, mean_block_us=total/blocks,
                       peak_block_us=peak, blocks_over_16ms=overruns, min_internal_heap=heap,
                       speex_allocated_internal=heap_before-heap_ready, uart_sha256=ch.sha(log))
            if opens:
                selected = samples(uart, "AUDIO", count)
                raw = mic[first:emitted]
                assert ch.digest(raw) == old["lanes"]["raw"]["input_sha256"]
                row["fifo_sha256"] = ch.digest(selected)
                log = work / "opus.log"
                payload = struct.pack("<3I", 0x3552504f, index, count) + selected.tobytes() + raw.tobytes()
                uplink.boot(uplink.CODEC, log, b"UPLINK PASS", payload, work)
                text = log.read_text()
                logs.write(log, f"{index:02}-opus.log")
                row["lanes"] = {}
                for lane, name in enumerate(("selected", "raw")):
                    packets = uplink.packets(text, index, count, lane)
                    pcm = uplink.decode(decoder, packets)
                    digest = ch.digest(pcm)
                    row["lanes"][name] = dict(packets=len(packets), decoded_sha256=digest)
                    if name == "raw":
                        assert digest == old["lanes"]["raw"]["decoded_sha256"]
                    else:
                        result = uplink.recognition(reader, pcm, meta["carrier"])
                        row["selected_asr"] = {k: result[k] for k in ("text", "complete_suffix", "exact")}
                        if index in paired:
                            other_lane = paired[index]["lanes"]["speex_p4"]
                            row["same_separate_fifo"] = row["fifo_sha256"] == other_lane["input_sha256"]
                            row["same_separate_opus_pcm"] = digest == other_lane["decoded_sha256"]
            records.append(row)
            print(index, row.get("same_separate_fifo"), row.get("selected_asr"), flush=True)
        assert seen == set(previous)
        failures = sorted(work.glob("*-attempt*.gz"))
        for failure in failures:
            logs.write(failure, failure.name)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for row in records:
            i = row["index"]
            if not row["open"]:
                continue
            text = logs.read(f"{i:02}-opus.log").decode().replace("\r\n", "\n")
            for lane, name in enumerate(("selected", "raw")):
                packets = uplink.packets(text, i, row["count"], lane)
                assert ch.digest(uplink.decode(decoder, packets)) == row["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    source = Path(__file__).parent / "main"
    ch.save(OUT, dict(records=records, asr=identity, model_sha256=ch.sha(model), qemu_sha256=fw["qemu_sha256"],
                      listener_sha256=ch.sha(app), listener_bytes=app.stat().st_size, baseline_bytes=old_app.stat().st_size,
                      elf_sha256=ch.sha(BUILD/"nyx_mac_listener.elf"), opus_sha256=fw["opus_sha256"],
                      archive_sha256=ch.sha(archive), retries=[p.name for p in failures],
                      sources={p.name:ch.sha(p) for p in (Path(__file__), fixed.OUT, other.OUT, source/"listener.cpp",
                                                          source/"CMakeLists.txt", src, uplink.listener.WORK/"sdkconfig")},
                      policy="SpeexDSP and the baseline ESP-SR plus AGC both run in the listening task. FIFO uses "
                      "the Speex output on synchronous stereo. Raw wake gates and FIFO geometry checked against "
                      "each original run. Actual FIFO sent to a separate P4 Opus image. Timing excludes UART "
                      "PCM dump, codec, physical I/O and whole-panel tasks."))


if __name__ == "__main__":
    main()
