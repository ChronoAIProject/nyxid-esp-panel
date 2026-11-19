"""Exercise the fixed reference switch and decode the matched P4 uplinks."""
import argparse
import io
import re
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import replay as listener
import transition_inputs as inputs
from check_c161_v3_two_anchor_v1 import binding
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import BUILD as CODEC, decode, decoder_api

BUILD = listener.WORK / "build"
LANES = (("selected", "raw"), ("processed", "delayed_raw"))


def samples(text, tag, count):
    data = bytearray()
    for at, hexbytes in re.findall(r"^"+tag+r" (\d+) ([0-9a-f]+)$", text, re.M):
        assert len(data) == 2*int(at)
        data.extend(bytes.fromhex(hexbytes))
    assert len(data) == count*2
    return np.frombuffer(data, dtype="<i2")


def weights(reference):
    # Independently express the hold as a deadline on the stream clock.
    last = -8001
    weight = 0
    result = np.empty(len(reference), dtype=np.int32)
    for t, x in enumerate(reference):
        if abs(int(x)) >= 64:
            last = t
        weight = min(160, weight+1) if t-last <= 8000 else max(0, weight-1)
        result[t] = weight
    return result


def cached_boot(build, log, marker, payload):
    if not log.exists() or marker.decode() not in log.read_text():
        listener.boot(build, log, log.parent/"flash.bin", marker, (2<<20, payload))


def analyze(text, row, pair, detector):
    i, length, chunk, prime = listener.values(text, "READY")
    emitted, first, count, blocks, total_us, peak_us, over, heap = listener.values(text, "FINISH")
    assert (i,length,chunk,prime) == (row["index"],len(pair),256,81600)
    assert blocks == (length+chunk-1)//chunk and emitted == blocks*chunk//160*160
    hits = [tuple(map(int, line.split())) for line in re.findall(r"^HITS (.+)$", text, re.M)]
    routed = [tuple(map(int, line.split())) for line in re.findall(r"^CUE_ROUTE (.+)$", text, re.M)]
    mic, ref = np.pad(pair, ((0,blocks*chunk-length),(0,0))).T
    cue = [e["t"] for e in detect(detector, [mic])]
    refs = [e["t"] for e in detect(detector, [ref])]
    assert cue == [v for _,_,v in hits if v >= 0]
    allowed = [t for t in cue if not any(0 <= t-r <= 3488 for r in refs)]
    assert routed == [(t, t if t in allowed else -1) for t in cue]
    if row["group"] == "regression":
        assert row["original_raw_v5"] == cue
    notifications = [block for block, c161, v5 in hits if c161>=0 or v5 in allowed]
    record = dict(index=i, c161=[t for _,t,_ in hits if t>=0], v5=cue, reference_v5=refs,
                  allowed_v5=allowed, first=first, emitted=emitted, count=count, blocks=blocks,
                  mean_block_us=total_us/blocks, peak_block_us=peak_us, over_budget=over, min_internal=heap)
    if not notifications:
        assert first == -1 and count == 0 and "OPEN " not in text
        return record, {}
    delivered, produced, opened = listener.values(text, "OPEN")
    assert delivered == notifications[0] and produced == delivered//160*160
    assert first == opened == max(0, produced-32000) and count == emitted-first
    raw = mic[first:emitted]
    delayed = np.pad(mic, (1440,0))[first:emitted]
    mix = weights(ref[:emitted])[first:]
    processed = samples(text,"PROCESSED",count)
    selected = samples(text,"AUDIO",count)
    numer = delayed*(160-mix) + processed*mix
    expected = np.sign(numer)*(np.abs(numer)//160)
    np.testing.assert_array_equal(selected, expected)
    lanes = dict(raw=raw, delayed_raw=delayed, processed=processed, selected=selected)
    record.update(delivered=delivered, command_lead=row["command_at"]-first,
                  capture_sha256={k:listener.digest(v) for k,v in lanes.items()},
                  samples_pure_raw=int((mix==0).sum()), samples_pure_processed=int((mix==160).sum()),
                  samples_fading=int(((mix>0)&(mix<160)).sum()))
    return record, lanes


def compatibility(logdir):
    plan = listener.read(listener.rooms.PLAN)["records"]
    old = listener.read(listener.RESULTS/(listener.STEM+".json"))["records"]
    # Four adjacent matched roles with the original mono flash protocol and zero reference.
    records = []
    for i in range(4):
        pcm = listener.load_wav(plan[i])
        log = logdir/f"mono-{i}.log"
        cached_boot(BUILD, log, b"LISTENER PASS", struct.pack("<3I",0x584e594d,i,len(pcm))+pcm.tobytes())
        text = log.read_text()
        emitted, first, count, *_ = listener.values(text, "FINISH")
        hits = [list(map(int, h.split())) for h in re.findall(r"^HITS (.+)$",text,re.M)]
        assert hits == old[i]["events"] and (emitted,first,count) == tuple(old[i][k] for k in ("emitted","first","count"))
        if count:
            assert listener.digest(samples(text,"AUDIO",count)) == old[i]["processed_sha256"]
        records.append(dict(index=i, events=hits, first=first, count=count, uart_sha256=listener.sha(log)))
    return records


def packets(text, index, count, lane):
    (section, n), = re.findall(rf"^UPLINK {index} {lane} {count}\n(.*?)^UPLINK_END {lane} (\d+)$",text,re.M|re.S)
    data = []
    for pts, start, packet in re.findall(r"^PACKET (\d+) (\d+) ([0-9a-f]+)$", section, re.M):
        assert (int(pts), int(start)) == (len(data)*20, len(data)*320)
        data.append(bytes.fromhex(packet))
    assert len(data) == int(n) == (count+319)//320
    return data


def main(action, source, output):
    plan = listener.read(source/inputs.PLAN)
    assert listener.sha(source/inputs.AUDIO) == plan["archive_sha256"]
    fw = BUILD/"nyx_mac_listener.bin"
    model = listener.WORK/"model/hey_nyx_model.bin"
    assert listener.sha(model) == listener.C161 and fw.read_bytes().count(model.read_bytes()) == 1
    versions = [listener.sha(p) for p in (fw, CODEC/"nyx_first_command.bin", source/inputs.PLAN)]
    root = inputs.WORK / ("-".join(h[:12] for h in versions))
    root.mkdir(exist_ok=True, parents=True)
    old = compatibility(root)
    rows = []
    if action == "score":
        reader, asr = listener.load_reader()
        lib = decoder_api()
    with TemporaryDirectory() as tmp, ZipFile(source/inputs.AUDIO) as archive:
        detector = binding(Path(tmp)/"v5.so",coded="v5")
        for row in plan["records"]:
            i = row["index"]
            pair, rate = sf.read(io.BytesIO(archive.read(f"{i:03}.flac")),dtype="int16")
            assert rate == 16000 and [listener.digest(v) for v in pair.T] == [row["mic_sha256"], row["ref_sha256"]]
            log = root/f"{i:03}.log"
            payload = struct.pack("<3I",0x5352594e,i,len(pair))+pair.T.astype("<i2").tobytes()
            cached_boot(BUILD,log,b"LISTENER PASS",payload)
            record, lanes = analyze(log.read_text(),row,pair,detector)
            record.update(uart_sha256=listener.sha(log), uplink={})
            for n, labels in enumerate(LANES if lanes else ()):
                count = record["count"]
                pair_id = 2*i+n
                payload = struct.pack("<3I",0x3552504f,pair_id,count)+b"".join(lanes[k].tobytes() for k in labels)
                uplink = root/f"{i:03}-opus-{n}.log"
                cached_boot(CODEC,uplink,b"UPLINK PASS",payload)
                if action == "score":
                    text = uplink.read_text()
                    for j,k in enumerate(labels):
                        data = packets(text, pair_id, count, j)
                        audio = decode(lib,data)
                        words = listener.transcripts(reader,audio)
                        got = listener.normalized(words["text"])
                        want = listener.normalized(row["carrier"])
                        record["uplink"][k] = dict(packets=len(data),bytes=sum(map(len,data)),
                            decoded_sha256=listener.digest(audio),asr=words,exact_suffix=got[-len(want):]==want,
                            contains_command=any(got[a:a+len(want)]==want for a in range(len(got))),
                            uart_sha256=listener.sha(uplink))
            rows.append(record)
            print(action,i,row["role"],record["first"],flush=True)
    if action == "score":
        output.mkdir(parents=True,exist_ok=True)
        logs = output/(inputs.STEM+"-uart.zip")
        with ZipFile(logs,"x",ZIP_DEFLATED,compresslevel=3) as z:
            for log in sorted(root.glob("*.log")):
                z.write(log,log.name)
        listener.save(output/(inputs.STEM+".json"),dict(records=rows, compatibility=old,
            listener_sha256=versions[0], codec_sha256=versions[1], plan_sha256=versions[2],
            app_bytes=fw.stat().st_size, qemu_sha256=listener.sha(listener.QEMU),
            mac_commit=listener.MAC, scorer_sha256=listener.sha(Path(__file__)), asr=asr,
            capture_sha256=listener.sha(listener.APP/"main/reference_capture.h"), uart_sha256=listener.sha(logs),
            opus=lib.opus_get_version_string().decode(), policy="Same P4 FIFO bounds. Four fresh P4 Opus encoders, whole decoded call, exact normalized command suffix. Host verifies routed v5 events and selected FIFO samples. C161 on raw mic independently opens FIFO."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("run","score"))
    parser.add_argument("--inputs",type=Path,default=inputs.WORK/"inputs")
    parser.add_argument("--out",type=Path,default=inputs.WORK/"score")
    args = parser.parse_args()
    main(args.action,args.inputs,args.out)
