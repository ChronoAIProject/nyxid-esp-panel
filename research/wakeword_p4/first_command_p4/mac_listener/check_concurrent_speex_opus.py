"""Replay fixed-v5 barge-in with a paced listener and concurrent P4 Opus worker in one guest."""
from hashlib import sha256
from pathlib import Path
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import torch

import check_joint_speex_capture as joint

ch, uplink = joint.ch, joint.uplink
BUILD = uplink.listener.WORK / "build-speex-opus"
OUT = ch.RESULTS / "4090-c161-concurrent-speex-opus.json"
ARCHIVE = OUT.with_suffix(".zip")


def packets(text):
    result = []
    for pts, start, data in re.findall(r"^MIC_PACKET (\d+) (\d+) ([a-f0-9]+)$", text, re.M):
        assert (int(pts), int(start)) == (20*len(result), 320*len(result))
        result.append(bytes.fromhex(data))
    return result


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    reader, asr = ch.load_reader()
    decoder = uplink.decoder_api()
    prev = ch.read(joint.OUT)
    assert prev["asr"] == asr
    fw = BUILD / "nyx_mac_listener.bin"
    c161 = (uplink.listener.WORK/"model/hey_nyx_model.bin").read_bytes()
    assert len(c161) == 60952 and sha256(c161).hexdigest() == prev["model_sha256"]
    assert fw.read_bytes().count(c161) == 1 and ch.sha(uplink.listener.QEMU) == prev["qemu_sha256"]
    originals = {(r["mic_sha256"],r["ref_sha256"]):r for r in prev["records"]
                 if r["source"]["mode"] in ("self_only","overlap")}
    work = uplink.listener.WORK / ("concurrent-" + ch.sha(fw)[:16])
    work.mkdir(exist_ok=True)
    uplink.WORK = work
    old_pack = joint.OUT.with_suffix(".zip")
    assert ch.sha(old_pack) == prev["archive_sha256"]
    rows, seen = [], set()
    with ZipFile(old_pack) as baseline, ZipFile(ARCHIVE.with_suffix(".tmp"),"w",ZIP_DEFLATED) as out:
        for meta, mic, ref in joint.fixed.spoken.extra_cases():
            key = ch.digest(mic), ch.digest(ref)
            if key not in originals or key in seen:
                continue
            seen.add(key)
            old = originals[key]
            i = old["index"]
            assert old["source"] == meta
            uart = work / f"{i:02}-{key[0][:12]}-{key[1][:12]}.log"
            if not uart.exists() or "LISTENER PASS" not in uart.read_text():
                payload = struct.pack("<3I",0x4452594e,i,len(mic)) + mic.tobytes() + ref.tobytes()
                uplink.boot(BUILD, uart, b"LISTENER PASS", payload, work)
            text = uart.read_text()
            assert uplink.listener.values(text,"READY") == [i,len(mic),256,81600]
            emitted, first, count, blocks, cost, peak, late, heap = uplink.listener.values(text,"FINISH")
            assert (emitted, first, count, blocks) == tuple(old[k] for k in ("emitted","first","count","blocks"))
            opened = list(map(int,re.findall(r"^OPEN (\d+) ",text,re.M)))
            allowed = sorted(map(int,re.findall(r"^(?:C161|V5)_ALLOW (\d+)$",text,re.M)))
            assert opened == old["open"] and allowed == old["allowed"]
            n, backlog, caught, pending, queued, worst, total, stack, before, ready, remaining = uplink.listener.values(text,"MIC_QUEUE")
            actual = packets(text)
            assert n == len(actual) == (count + 319)//320 and remaining == 0
            record = dict(index=i, source=meta, mic_sha256=key[0], ref_sha256=key[1], first=first, count=count,
                          open=opened, allowed=allowed, blocks=blocks, blocks_over_16ms=late, peak_block_us=peak,
                          mean_block_us=cost/blocks, min_internal_heap=heap, max_pending_input_blocks=pending,
                          blocks_started_with_backlog=queued, peak_fifo_samples=backlog, catch_up_us=caught,
                          peak_encode_us=worst, total_encode_us=total, mic_task_stack_free=stack,
                          internal_free_before_mic_task=before, internal_free_after_opus_open=ready,
                          packets=n, uart_sha256=ch.sha(uart), uart=uart.name)
            if opened:
                expected = uplink.packets(baseline.read(f"{i:02}-opus.log").decode().replace("\r\n","\n"),i,count,0)
                assert actual == expected and caught >= 0
                decoded = uplink.decode(decoder,actual)
                assert ch.digest(decoded) == old["lanes"]["selected"]["decoded_sha256"]
                reading = uplink.recognition(reader,decoded,meta["carrier"])
                assert reading["text"] == old["selected_asr"]["text"]
                record.update(decoded_sha256=ch.digest(decoded), asr=reading, lead_samples=meta["command_at"]-opened[0])
            else:
                assert caught == -1 and not n
            rows.append(record)
            out.write(uart,uart.name)
            print(i,n,caught,heap,flush=True)
        assert seen == set(originals) and len(rows) == 14
        for f in sorted(work.glob("*-attempt*.gz")):
            out.write(f,f.name)
    with ZipFile(ARCHIVE.with_suffix(".tmp")) as z:
        assert z.testzip() is None
        for r in rows:
            data = z.read(r["uart"])
            assert sha256(data).hexdigest() == r["uart_sha256"]
            if r["packets"]:
                audio = uplink.decode(decoder,packets(data.decode().replace("\r\n","\n")))
                assert ch.digest(audio) == r["decoded_sha256"]
    ARCHIVE.with_suffix(".tmp").replace(ARCHIVE)
    project = Path(__file__).parent
    codec = Path(ch.read(BUILD/"project_description.json")["build_component_info"]["esp_audio_codec"]["dir"])
    sources = [project/"CMakeLists.txt", project/"main/CMakeLists.txt", project/"main/capture_opus.h",
               project/"main/listener.cpp", Path(__file__), codec/"idf_component.yml",
               uplink.listener.WORK/"sdkconfig", joint.OUT, old_pack]
    ch.save(OUT, dict(records=rows, images={f:ch.sha(BUILD/f) for f in ch.read(BUILD/"flasher_args.json")["flash_files"].values()},
            app_bytes=fw.stat().st_size, app_sha256=ch.sha(fw), elf_sha256=ch.sha(BUILD/"nyx_mac_listener.elf"),
            model_sha256=prev["model_sha256"], qemu_sha256=prev["qemu_sha256"], asr=asr,
            separate_listener_bytes=prev["listener_bytes"], archive_sha256=ch.sha(ARCHIVE),
            sources={str(p):ch.sha(p) for p in sources}, retries=[p.name for p in sorted(work.glob("*-attempt*.gz"))],
            policy="ESP32-P4 QEMU. Pace 256 samples every 16 ms with esp_timer. At signed-reference OPEN start "
            "a second FreeRTOS task with the production encoder settings and 48 KiB stack, 320 samples per frame, "
            "four-frame bursts. Immediately ready, locally buffered packet sink. Keep both tasks and detectors "
            "running through end of replay; pad only the final partial frame. Save packets after the clock stops. "
            "Compare every byte to the prior separate P4 Opus run and re-decode UART. Virtual I2S, no transport, "
            "provider wait, UI, radio or hardware. Internal heap includes fixture and scheduler instrumentation."))


if __name__ == "__main__":
    main()
