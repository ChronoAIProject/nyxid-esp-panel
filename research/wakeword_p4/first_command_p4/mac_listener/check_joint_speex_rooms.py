"""Check the joint capture on paired household commands with a continuously playing or interrupted panel."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np

import check_joint_speex_capture as joint
import handoff_barge_in as rooms
from transition import samples

ch, uplink = joint.ch, joint.uplink
OUT = ch.RESULTS / "4090-c161-joint-speex-rooms.json"


def fixtures(baseline):
    for i, (scene, prev) in enumerate(zip(rooms.scenes(), baseline["records"], strict=True)):
        meta, dry, panel, response = scene
        assert meta == prev["source"]
        noise = np.random.default_rng(18221).standard_normal(len(panel))
        for arm, old in (("continue", prev["continue"]), ("interrupt", prev["interrupt"])):
            ref = panel.copy()
            if arm == "interrupt":
                stop, = prev["continue"]["opened"]
                fade = rooms.FADE
                ref[stop:stop+fade] = np.rint(ref[stop:stop+fade] * np.arange(fade-1, -1, -1) / fade)
                ref[stop+fade:] = 0
            mic, adc, clips = rooms.render(dry, ref, response, meta["echo_gain"], noise)
            assert (ch.digest(mic), ch.digest(adc), clips) == (old["mic_sha256"], old["reference_sha256"], old["clipped_samples"])
            idx = 2*i + int(arm == "interrupt")
            yield idx, arm, meta, mic, adc, old


def summarize(reader, audio, command):
    r = uplink.recognition(reader, audio, command)
    return {k: r[k] for k in ("text", "complete_suffix", "exact")}


def main():
    base_file = rooms.BASE.with_suffix(".json")
    base_archive = rooms.BASE.with_suffix(".zip")
    baseline = ch.read(base_file)
    prior = ch.read(joint.OUT)
    assert ch.sha(base_archive) == baseline["archive_sha256"]
    assert ch.sha(joint.BUILD/"nyx_mac_listener.bin") == prior["listener_sha256"]
    assert ch.sha(uplink.CODEC/"nyx_first_command.bin") == prior["opus_sha256"] == baseline["codec_sha256"]
    assert ch.sha(uplink.listener.QEMU) == prior["qemu_sha256"] == baseline["qemu_sha256"]
    small, identity = ch.load_reader()
    assert identity == baseline["asr"] == prior["asr"]
    asr = joint.other.stop.audit
    turbo_files = ch.read(ch.RESULTS / (asr.STEM + ".json"))["turbo"]["files"]
    assert all(ch.sha(asr.TURBO/f) == sha for f, sha in turbo_files.items())
    turbo = WhisperModel(str(asr.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = uplink.decoder_api()
    paths = OUT.with_suffix(".zip")
    records = []
    with TemporaryDirectory() as tmp, ZipFile(base_archive) as previous, ZipFile(paths.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        for idx, arm, meta, mic, adc, old in fixtures(baseline):
            # Same third-anchor and frame-activity protocol as the original room run.
            log = work / "listener.log"
            payload = struct.pack("<3I", 0x4652594e, idx, len(mic)) + mic.tobytes() + adc.tobytes()
            uplink.boot(joint.BUILD, log, b"LISTENER PASS", payload, work)
            text = log.read_text()
            event_types = ("READY ", "HITS ", "OPEN ", "CUE_ROUTE ", "C161_ALLOW ", "V5_ALLOW ")
            actual = [s for s in text.splitlines() if s.startswith(event_types)]
            assert actual == [s for s in old["uart_events"] if s.startswith(event_types)]
            emitted, first, count, blocks, total, peak, late, heap = uplink.listener.values(text, "FINISH")
            assert [emitted, first, count, blocks] == uplink.listener.values("\n".join(old["uart_events"]), "FINISH")[:4]
            opened, = old["opened"]
            assert count == old["uplink"]["samples"] and meta["cue_at"] <= opened < meta["command_at"]
            selected = samples(text, "AUDIO", count)
            # The codec gets the exact joint FIFO and raw captured mic at the same interval.
            raw = np.pad(mic, (0, max(0, emitted-len(mic))))[first:emitted]
            assert selected.shape == raw.shape
            lines = [s for s in text.splitlines() if s.startswith(event_types + ("FINISH ", "SPEEX_HEAP ", "LISTENER PASS"))]
            logs.writestr(f"{idx:02}-events.txt", "\n".join(lines) + "\n")
            log = work/"codec.log"
            payload = struct.pack("<3I", 0x3552504f, idx, count) + selected.tobytes() + raw.tobytes()
            uplink.boot(uplink.CODEC, log, b"UPLINK PASS", payload, work)
            uart = log.read_text()
            logs.write(log, f"{idx:02}-opus.log")
            prev_uart = previous.read(f"{idx:02}-opus.log").decode().replace("\r\n","\n")
            cut = opened - first - 1600
            row = dict(index=idx, arm=arm, source=meta, mic_sha256=old["mic_sha256"], ref_sha256=old["reference_sha256"],
                       opened=opened, first=first, count=count, blocks=blocks, mean_block_us=total/blocks,
                       peak_block_us=peak, blocks_over_16ms=late, min_internal_heap=heap, stages={})
            for name, source, lane in (("baseline", prev_uart, 0), ("speex", uart, 0), ("raw", uart, 1)):
                packets = uplink.packets(source, idx, count, lane)
                audio = uplink.decode(decoder, packets)
                digest = ch.digest(audio)
                if name == "baseline":
                    assert digest == old["uplink"]["pcm_sha256"]
                if name == "raw":
                    assert packets == uplink.packets(prev_uart, idx, count, 1)
                result = dict(packets=len(packets), decoded_sha256=digest)
                for view, signal in (("full", audio), ("open_minus_100ms", audio[cut:])):
                    result[view] = {k: summarize(reader, signal, meta["carrier"]) for k, reader in (("small.en", small), ("large-v3-turbo", turbo))}
                if name == "baseline":
                    assert result["full"]["small.en"]["text"] == old["uplink"]["recognition"]["text"]
                row["stages"][name] = result
            row["speex_fifo_sha256"] = ch.digest(selected)
            records.append(row)
            print(idx, arm, row["stages"]["speex"]["full"]["small.en"]["text"], flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for p in failures:
            logs.write(p, p.name)
    assert len(records) == 32
    with ZipFile(paths.with_suffix(".tmp")) as pack:
        assert pack.testzip() is None
        for r in records:
            uart = pack.read(f"{r['index']:02}-opus.log").decode().replace("\r\n","\n")
            for i, name in enumerate(("speex", "raw")):
                packets = uplink.packets(uart, r["index"], r["count"], i)
                assert ch.digest(uplink.decode(decoder, packets)) == r["stages"][name]["decoded_sha256"]
    paths.with_suffix(".tmp").replace(paths)
    ch.save(OUT, dict(records=records, asr=identity, turbo_files=turbo_files, archive_sha256=ch.sha(paths), retries=[p.name for p in failures],
                      sources={p.name: ch.sha(p) for p in (base_file, base_archive, joint.OUT, Path(__file__))},
                      policy="Rebuild consumed kitchen and living-room input pairs with the archived line reference. "
                      "Preserve the original OPEN and FIFO bounds for the two captured arms. Full two-second lookback "
                      "and open minus 100 ms on real P4 Opus, two unprompted GPU readers. Four voices and two commands "
                      "repeated across rooms. QEMU timing covers the joint listening task."))


if __name__ == "__main__":
    main()
