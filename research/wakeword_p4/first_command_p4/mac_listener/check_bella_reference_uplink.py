"""Restore the panel's digital R, then compare matched P4 capture paths at the same wake."""
from hashlib import sha256
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_product_uplink as prior
import replay_bella_command_prefix as bella

P = prior.replay
OUT = P.RESULTS / "4090-c161-bella-reference-uplink.json"
WORK = Path(".local/wakeword/c161-bella-reference-uplink")
BUILD = Path(".local/wakeword/c161-panel-middle-uart/build")
TRACE = WORK / "4090-c161-bella-reference-uplink-uart.zip"


def verified_boot(work, archive, member, build, binary, marker, payload, inspect):
    # Retain the raw diagnostics and reuse only a complete, parsed transfer of this same input.
    key = sha256(bytes.fromhex(P.sha(build/binary)) + payload).hexdigest()
    cache = WORK/"verified"
    cache.mkdir(exist_ok=True)
    log = cache/(key+".log")
    for attempt in range(3):
        try:
            if not log.exists():
                P.boot(build, log, work/"flash.bin", marker, (2<<20,payload))
            text = log.read_text()
            assert marker.decode() in text
            inspect(text)
            break
        except (AssertionError, TimeoutError, ValueError):
            if log.exists():
                log.replace(cache/f"{key}.failed-{attempt}.log")
            if attempt == 2:
                raise
    for failure in sorted(cache.glob(key+".failed-*.log")):
        archive.write(failure, f"retry/{member}/{failure.name}")
    archive.write(log,member)
    return text, P.sha(log)


def listen(work, archive, name, i, mic, ref):
    payload = struct.pack("<3I", 0x5854594e, i, len(mic)) + mic.tobytes() + ref.tobytes() + bytes(4)
    def inspect(text):
        _, _, count, *_ = P.values(text,"FINISH")
        if count:
            for tag in ("PROCESSED","AUDIO"):
                prior.samples(text,tag,count)
    text, uart_sha = verified_boot(work,archive,f"{i:02}/{name}.log",BUILD,"nyx_mac_listener.bin",
                                   b"LISTENER PASS",payload,inspect)
    assert P.values(text, "READY") == [i, len(mic), 256, 81600]
    open_read = [int(t) for t in re.findall(r"^OPEN (\d+) ", text, re.M)]
    return text, dict(uart_sha256=uart_sha, watchdog_warnings=text.count("Task watchdog got triggered"), reference_sha256=P.digest(ref), opened=open_read,
                      finish=P.values(text,"FINISH"), allowed_cue=[int(t) for t in re.findall(r"^V5_ALLOW (\d+)$",text,re.M)],
                      allowed_c161=[int(t) for t in re.findall(r"^C161_ALLOW (\d+)$",text,re.M)])


def encode(work, archive, i, name, a, b, expected, gpu):
    assert len(a) == len(b) and len(a) > 0
    data = struct.pack("<3I", 0x3552504f, i, len(a)) + a.tobytes() + b.tobytes()
    def inspect(text):
        for lane in (0,1):
            prior.packets(text,i,len(a),lane)
    text, uart_sha = verified_boot(work,archive,f"{i:02}/{name}.log",prior.CODEC,"nyx_first_command.bin",
                                   b"UPLINK PASS",data,inspect)
    proc, model, turbo, decoder = gpu
    records = []
    for lane, pcm in enumerate((a,b)):
        packets = prior.packets(text,i,len(pcm),lane)
        audio = prior.decode(decoder, packets)
        records.append(dict(uart_sha256=uart_sha, watchdog_warnings=text.count("Task watchdog got triggered"),
                            fifo_sha256=P.digest(pcm), decoded_sha256=P.digest(audio),
                            packets=len(packets), bytes=sum(map(len,packets)), samples=len(pcm),
                            qwen=prior.evaluate(proc,model,audio,expected),
                            turbo=prior.recognition(turbo,audio,expected)))
    return records


def main():
    base, earlier = P.read(prior.BASE), P.read(prior.OUT)
    assert P.sha(prior.BASE) == earlier["sources"][prior.BASE.name]
    firmware = P.sha(BUILD/"nyx_mac_listener.bin")
    assert firmware == earlier["capture_firmware"]
    assert P.sha(P.QEMU) == base["codec"]["qemu_sha256"]
    assert P.sha(prior.CODEC/"nyx_first_command.bin") == base["codec"]["opus_sha256"]
    panel_path = bella.study.mixed.panel.demo.WAVE
    panel, rate = sf.read(panel_path, dtype="int16")
    assert rate == 16000 and panel.ndim == 1
    proc, model, qwen = prior.load()
    assert qwen == base["qwen"]
    gpu = proc, model, prior.reader(base["turbo"]), prior.decoder_api()
    records = []
    WORK.mkdir(parents=True,exist_ok=True)
    cases = (v for v in bella.recordings() if v[0]["paired_back_grid"])
    with TemporaryDirectory(dir=WORK) as folder:
        work = Path(folder)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as raw, ZipFile(work/OUT.with_suffix(".zip").name,"w",ZIP_DEFLATED) as opus:
            for (parent, source, mic, _, _), missing, old in zip(cases,earlier["records"],base["captures"],strict=True):
                i = len(records)
                assert source == old["source"] and old["mic_sha256"] == P.digest(mic)
                assert i == missing["index"] == old["index"] and parent["batch"] == old["batch"]
                ref = np.zeros_like(mic)
                at = source["panel_at"]
                ref[at:at+len(panel)] = panel
                full_log, full = listen(work, raw, "continuous", i, mic, ref)
                row = dict(index=i, file=missing["file"], command=source["command"], command_at=source["command_at"],
                           db=source["external_db"], panel_at=at, mic_sha256=P.digest(mic), continuous=full,
                           missing_r_open=missing["opened"])
                if full["opened"]:
                    opened, = full["opened"]
                    emitted, first, count = full["finish"][:3]
                    assert first == max(0, opened//160*160-32000) and count == emitted-first
                    # Board marker always sees physical R. Only the product AEC masks R during idle.
                    # The read that triggered the call is still idle. From the next read it sees R.
                    gated = ref.copy()
                    gated[:opened] = 0
                    masked_log, masked = listen(work,raw,"from-call",i,mic,gated)
                    # This second pass supplies DSP samples. Its internal wake is allowed to occur
                    # earlier. Align the output to the proven full-reference decision above.
                    end2, first2, count2 = masked["finish"][:3]
                    assert 0 <= first2 <= first and end2 == emitted and count2 == end2-first2
                    product = prior.samples(masked_log,"PROCESSED",count2)[first-first2:]
                    continuous = prior.samples(full_log,"PROCESSED",count)
                    selected = prior.samples(full_log,"AUDIO",count)
                    # Reconstruct the delayed-raw control on the same output clock.
                    padded_mic = np.pad(mic,(1440,full["finish"][3]*256-len(mic)))
                    delayed = padded_mic[first:emitted]
                    assert len(product) == len(continuous) == len(selected) == len(delayed) == count
                    uplink1 = encode(work,opus,i,"aec",product,continuous,row["command"],gpu)
                    uplink2 = encode(work,opus,i,"reference-select",selected,delayed,row["command"],gpu)
                    row.update(masked_dsp=masked, timely=source["start"] <= opened < source["command_at"],
                               product=uplink1[0], continuous_aec=uplink1[1], reference_select=uplink2[0], delayed_raw=uplink2[1])
                records.append(row)
                print(i+1, "of 24", full["opened"], flush=True)
        assert len(records) == len({r["file"] for r in records}) == 24
        archive = OUT.with_suffix(".zip")
        P.save(work/OUT.name, dict(records=records, qwen=qwen, turbo=base["turbo"], codec=base["codec"],
               model=base["model"], firmware=firmware, board_base=P.MAC, panel_sha256=P.sha(panel_path),
               sources={p.name:P.sha(p) for p in (prior.BASE, prior.OUT, Path(__file__), Path(bella.__file__))},
               archive_sha256=P.sha(work/archive.name), uart_archive=TRACE.name, uart_sha256=P.sha(work/TRACE.name),
               policy="24 existing rear room mixes. Synthetic electrical reference equals the distributed quiet "
                      "panel WAV at its original playout sample. No independent TX event. Wake uses full R. "
                      "Hold the resulting open, full two-second history and end constant. Compare AEC/AGC receiving R "
                      "from the first read after OPEN or from the first listen read, reference-selected PCM, and delayed "
                      "raw. Independent cold-start P4 24 kb/s Opus per lane. Complete decoded FIFO on RTX 4090."))
        (work/TRACE.name).replace(TRACE)
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
