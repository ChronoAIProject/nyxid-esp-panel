"""Exercise the Mac wake sender under immediate readiness and one fixed connection wait."""
import argparse
from hashlib import sha256
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import check_bella_reference_uplink as reference

P, prior = reference.P, reference.prior
WORK = Path(".local/wakeword/c161-bella-sender")
BUILD = WORK / "build"
BOARD = Path(".local/wakeword/c161-mac-board-marker/src/components/panel_services")
APP = P.APP.parent / "mac_board_marker"
OUT = P.RESULTS / "4090-c161-bella-sender.json"


def archived(archive, member, digest):
    raw = archive.read(member)
    assert sha256(raw).hexdigest() == digest
    return raw.decode().replace("\r\n","\n")


def load_cases():
    baseline = P.read(reference.OUT)
    assert P.sha(reference.TRACE) == baseline["uart_sha256"]
    assert P.sha(reference.OUT.with_suffix(".zip")) == baseline["archive_sha256"]
    with ZipFile(reference.TRACE) as captures, ZipFile(reference.OUT.with_suffix(".zip")) as codec:
        for i, row in enumerate(baseline["records"]):
            assert i == row["index"] and row["timely"]
            product, masked = row["product"], row["masked_dsp"]
            end, first, samples = row["continuous"]["finish"][:3]
            opened, = row["continuous"]["opened"]
            assert first + 32000 == opened//160*160 and samples == end-first == product["samples"]
            end2, first2, count2 = masked["finish"][:3]
            assert end2 == end and 0 <= first2 <= first
            trace = archived(captures, f"{i:02}/from-call.log", masked["uart_sha256"])
            pcm = prior.samples(trace, "PROCESSED", count2)[first-first2:]
            assert P.digest(pcm) == product["fifo_sha256"]
            uart = archived(codec, f"{i:02}/aec.log", product["uart_sha256"])
            packets = prior.packets(uart, i, samples, 0)
            yield row, struct.pack("<4I", i, opened, first, samples) + pcm.tobytes(), packets


def compare(log, i, opened, first, count, original):
    text = log.read_text()
    n = len(original)
    assert text.count("SENDER PASS") == 1 and text.count("BEGIN ") == 2
    captures = []
    for lane in range(2):
        start = f"BEGIN {i} {lane} {opened} {first} {count}\n"
        body, = re.findall("^" + start + r"(.*?)^END " + str(lane) + r" .*?$", text, re.S | re.M)
        entries = [(how, int(at), bytes.fromhex(packet)) for how, at, packet in
                   re.findall(r"^(SENT|UNSENT) (\d+) ([0-9a-f]+)$", body, re.M)]
        dropped, silence = (25, 50) if lane else (0, 0)
        encoded, sent, padded, peak = map(int, re.search(rf"^END {lane} (\d+) (\d+) (\d+) (\d+)$", text, re.M).groups())
        assert encoded == len(entries) == n + dropped + silence and sent == n + silence and padded == n * 320
        assert [(how, at) for how, at, _ in entries] == [
            ("UNSENT" if k < dropped else "SENT", k*20) for k in range(encoded)]
        frames = [p for _, _, p in entries[dropped:]]
        epoch = lane+1
        ready_ms = 1500 if lane else 0
        assert re.findall(r"^first_pcm (.*)$",body,re.M) == [
            f"epoch={epoch} start={first} samples=320 input_us={1000*ready_ms}"]
        sends = re.findall(r"^peer_send (.*)$",body,re.M)
        expected = []
        if lane:
            expected.append(f"epoch=2 first=1 captured=0 start=-1 pts=500 bytes={len(frames[0])} accepted_us=500000")
        expected.append(f"epoch={epoch} first={1-lane} captured=1 start={first} pts={ready_ms} "
                        f"bytes={len(frames[silence])} accepted_us={1000*ready_ms}")
        assert sends == expected
        if not lane:
            assert frames == original
        captures.append((frames, dict(unsent=dropped, sent_silence=silence, captured=n,
                        first_pcm_sample=first, first_send_pts=20*dropped, first_captured_pts=ready_ms,
                        replayed_samples=padded, terminal_padding=padded-count, peak_fifo_samples=peak)))
    return captures


def run_case(work, payload, original):
    index, opened, first, count = struct.unpack_from("<4I", payload)
    key = sha256(bytes.fromhex(P.sha(BUILD/"board.bin")) + payload).hexdigest()
    log = work / (key + ".log")
    if not log.exists() or b"SENDER PASS\r\n" not in log.read_bytes():
        if log.exists():
            log.rename(work / (key + ".incomplete-" + P.sha(log)[:12] + ".log"))
        P.boot(BUILD, log, work / "flash.bin", b"SENDER PASS", (2<<20,payload))
    captures = compare(log, index, opened, first, count, original)
    return log, captures


def collect():
    base = P.read(reference.OUT)
    assert P.sha(P.QEMU) == base["codec"]["qemu_sha256"]
    for row, payload, cold in load_cases():
        log, captures = run_case(WORK, payload, cold)
        yield row, log, captures


def score():
    base = P.read(reference.OUT)
    proc, model, qwen = prior.load()
    assert qwen == base["qwen"]
    turbo = prior.reader(base["turbo"])
    decoder = prior.decoder_api()
    rows = []
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        with ZipFile(temp/archive.name, "w", ZIP_DEFLATED) as logs:
            for row, log, ((cold, immediate), (warm, waited)) in collect():
                cold_audio, warm_audio = (prior.decode(decoder, stream) for stream in (cold, warm))
                product = row["product"]
                assert P.digest(cold_audio) == product["decoded_sha256"] == product["qwen"]["pcm_sha256"]
                prefix = waited["sent_silence"]
                leading, tail = warm_audio[:320*prefix], warm_audio[320*prefix:]
                assert len(tail) == len(cold_audio)
                readings = dict(immediate=dict(qwen=product["qwen"], turbo=product["turbo"]))
                conditions = dict(waited=warm_audio, waited_trimmed=tail, immediate_prefixed=np.r_[leading,cold_audio])
                for name, audio in conditions.items():
                    readings[name] = dict(qwen=prior.evaluate(proc,model,audio,row["command"]),
                                          turbo=prior.recognition(turbo,audio,row["command"]))
                i = row["index"]
                logs.write(log, f"{i:02}.log")
                rows.append(dict(index=i, file=row["file"], command=row["command"], db=row["db"],
                                 command_at=row["command_at"], opened=row["continuous"]["opened"][0],
                                 pcm_sha256=product["fifo_sha256"], uart_sha256=P.sha(log),
                                 watchdog_warnings=log.read_text().count("Task watchdog got triggered"),
                                 immediate=immediate, waited=waited, readings=readings,
                                 prefix_decoded_peak=int(np.max(np.abs(leading.astype(np.int32)))),
                                 changed_captured_packets=sum(a!=b for a,b in zip(cold,warm[prefix:],strict=True)),
                                 changed_decoded_samples=int(np.count_nonzero(cold_audio!=tail))))
                print(f"Scored {i+1}/24", flush=True)
        assert len(rows) == len({r["file"] for r in rows}) == 24
        counts = {}
        for case in rows[0]["readings"]:
            counts[case] = {}
            for reader, key in (("qwen","complete"),("turbo","complete_suffix")):
                pairs = [(r["readings"]["immediate"][reader][key],r["readings"][case][reader][key]) for r in rows]
                counts[case][reader] = dict(complete=sum(b for _,b in pairs), gained=pairs.count((False,True)),
                                            lost=pairs.count((True,False)), total=len(rows))
        files = [Path(__file__), APP/"build_guest.py", APP/"build_sender.py", APP/"sender.c",
                 APP/"c161-fixed-marker.patch", BOARD/"panel_voice.c", BOARD/"panel_voice_audio.c",
                 BUILD/"mic_wake.inc", BUILD/"board.bin", reference.OUT, reference.TRACE,
                 reference.OUT.with_suffix(".zip"), prior.CODEC.parent/"esp_audio_codec/lib/esp32p4/libesp_audio_codec.a"]
        P.save(temp/OUT.name, dict(records=rows, counts=counts, board_base=P.MAC, model=base["model"],
               qemu_sha256=P.sha(P.QEMU), sources={p.name:P.sha(p) for p in files},
               archive_sha256=P.sha(temp/archive.name), qwen=qwen, turbo=base["turbo"],
               decoder=decoder.opus_get_version_string().decode(),
               policy="24 consumed synthetic rear-room captures with panel and music. Use each original Nyx open "
                      "and the archived AEC/AGC FIFO. Execute the patched board mic_task wake branch, gate, FIFO "
                      "and P4 Opus. Single-threaded simulated 256-sample reads, 160-sample AGC blocks, nominal "
                      "10/20 ms sleeps, zero-time bursts, unmuted, logical clock starts at mic_task entry at OPEN. "
                      "Pad only the final partial frame with zeros as in the archived codec. Immediately connected "
                      "and ready versus connected at 500 ms and provider ready at 1500 ms. Peer accepts every send. "
                      "Compare all immediate packets against the archived bytes. Fresh decoder per sent stream. "
                      "waited_trimmed drops the decoded prefix after decoding. immediate_prefixed concatenates "
                      "that same decoded prefix with the immediate PCM. No controller, network, receiver VAD, "
                      "jitter, workload timing, resource or hardware measurements."))
        (temp/archive.name).replace(archive)
        (temp/OUT.name).replace(OUT)
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect","score"])
    if parser.parse_args().action == "collect":
        for row, _, _ in collect():
            print(f"Verified {row['index']+1}/24", flush=True)
    else:
        score()
