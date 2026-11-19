"""Replay the frozen fixed cue with concurrent spoken Bot audio on the P4 research listener and Opus."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import check_spoken_playback_echo as spoken
import coded_command_uplink as uplink
from reference_signature import qualify
from replay_c161_pcm_cues_v1 import detect
from self_guard import expected
from transition import samples, weights

ch = spoken.ch
OUT = ch.RESULTS / "4090-c161-fixed-cue-over-spoken-bot.json"


def run_case(work, record, mic, ref, index, detector, prefix, reader, decoder, logs):
    source = record["source"]
    log = work / "listener.log"
    payload = struct.pack("<3I", 0x4452594e, index, len(mic)) + mic.tobytes() + ref.tobytes()
    uplink.boot(uplink.BUILD, log, b"LISTENER PASS", payload, work)
    uart = log.read_text()
    assert uplink.listener.values(uart, "READY") == [index, len(mic), 256, 81600]
    tags = ("HITS ", "OPEN ", "V5_ALLOW ", "C161_ALLOW ", "CUE_ROUTE ", "FINISH ", "READY ", "LISTENER PASS")
    trace = [line for line in uart.splitlines() if line.startswith(tags)]
    logs.writestr(f"{index:02}-events.txt", "\n".join(trace) + "\n")
    hits = [list(map(int, line.split()[1:])) for line in trace if line.startswith("HITS ")]
    refs = [e["t"] for e in detect(detector, [ref])]
    signed = [e["t"] for e in detect(prefix, [ref]) if qualify(ref, e["t"])]
    allowed = []
    for name, col in (("c161", 1), ("v5", 2)):
        raw_events = [h[col] for h in hits if h[col] >= 0]
        accepted = [int(line.split()[1]) for line in trace if line.startswith(name.upper() + "_ALLOW ")]
        assert raw_events == record[name] and accepted == expected(raw_events, refs, ref, signed)
        allowed.extend(accepted)
    opens = [list(map(int, line.split()[1:])) for line in trace if line.startswith("OPEN ")]
    emitted, first, count, blocks, total_us, peak_us, late, heap = uplink.listener.values(uart, "FINISH")
    assert blocks == (len(mic) + 255) // 256 and emitted == blocks * 256 // 160 * 160
    assert [t[0] for t in opens] == ([(min(allowed) + 255) // 256 * 256] if allowed else [])
    row = dict(index=index, source=source, mic_sha256=ch.digest(mic), ref_sha256=ch.digest(ref),
               c161=record["c161"], v5=record["v5"], reference_matches=refs, reference_prefix=signed,
               allowed=sorted(allowed), open=[t[0] for t in opens], emitted=emitted, first=first, count=count,
               listener_uart_sha256=ch.sha(log), mean_block_us=total_us/blocks, peak_block_us=peak_us,
               blocks_over_16ms=late, min_internal_heap=heap, lanes={})
    if not opens:
        assert (first, count) == (-1, 0)
        return row
    opened, produced, captured = opens[0]
    assert captured == first == max(0, produced - 32000) and produced == opened // 160 * 160
    assert count == emitted - first
    selected = samples(uart, "AUDIO", count)
    processed = samples(uart, "PROCESSED", count)
    raw = mic[first:emitted]
    dry = np.pad(mic, (1440, 0))[first:emitted]
    w = weights(ref)[first:emitted]
    value = dry * (160 - w) + processed * w
    np.testing.assert_array_equal(selected, np.sign(value) * (np.abs(value) // 160))
    row.update(command_lead_samples=source["command_at"] - opened,
               pure_processed=int(np.count_nonzero(w == 160)), pure_raw=int(np.count_nonzero(w == 0)))
    codec = work / "opus.log"
    payload = struct.pack("<3I", 0x3552504f, index, count) + selected.tobytes() + raw.tobytes()
    uplink.boot(uplink.CODEC, codec, b"UPLINK PASS", payload, work)
    logs.write(codec, f"{index:02}-opus.log")
    output = codec.read_text()
    cut = max(0, opened - first - 1600)
    for i, (name, signal) in enumerate((("selected", selected), ("raw", raw))):
        packets = uplink.packets(output, index, count, i)
        pcm = uplink.decode(decoder, packets)
        lane = dict(input_sha256=ch.digest(signal), decoded_sha256=ch.digest(pcm), packets=len(packets))
        for view, offset in (("full", 0), ("from_open_minus_100ms", cut)):
            reading = uplink.recognition(reader, pcm[offset:], source["carrier"])
            lane[view] = {k: reading[k] for k in ("text", "complete_suffix", "exact")}
        row["lanes"][name] = lane
    return row


def main():
    firmware = uplink.firmware()
    reader, identity = ch.load_reader()
    decoder = uplink.decoder_api()
    frozen = {(r["mic_sha256"], r["ref_sha256"]): r for r in ch.read(spoken.OUT)["pairs"]
              if r["group"] == "additional_pairs"}
    records, seen = [], set()
    pack = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(pack.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        detector = ch.binding(work / "cue.so", coded="v5")
        prefix = ch.binding(work / "prefix.so", coded="aligned")
        for meta, mic, ref in spoken.extra_cases():
            if meta.get("role") != "fixed-v5" and meta["mode"] != "self_only":
                continue
            pair = (ch.digest(mic), ch.digest(ref))
            if pair in seen:
                continue
            seen.add(pair)
            record = frozen[pair]
            row = run_case(work, record, mic, ref, len(records), detector, prefix, reader, decoder, logs)
            records.append(row)
            print(len(records), meta["mode"], row["open"], flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for failure in failures:
            logs.write(failure, failure.name)
    assert len(records) == 26
    with ZipFile(pack.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for r in records:
            if not r["lanes"]:
                continue
            index = r["index"]
            uart = logs.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
            for i, name in enumerate(("selected", "raw")):
                packets = uplink.packets(uart, index, r["count"], i)
                assert ch.digest(uplink.decode(decoder, packets)) == r["lanes"][name]["decoded_sha256"]
    pack.with_suffix(".tmp").replace(pack)
    ch.save(OUT, dict(records=records, asr=identity, **firmware, archive_sha256=ch.sha(pack),
                      sources={p.name: ch.sha(p) for p in (Path(__file__), spoken.OUT, Path(spoken.__file__))},
                      retries=[p.name for p in failures], protocol="Signed playback-reference prefix listener. "
                      "Feed synchronous stereo to P4 QEMU. Read actual OPEN and selected FIFO, independently "
                      "check both wake gates and the reference-driven PCM crossfade. Encode selected and raw "
                      "in P4 Opus. Decode both lanes twice. Compare the unprompted small.en transcript suffix "
                      "with the labelled external command. Self-only controls deduplicated by both input hashes."))


if __name__ == "__main__":
    main()
