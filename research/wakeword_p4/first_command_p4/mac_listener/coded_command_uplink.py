"""Compare command gain and wake-clock cropping on the exact four-voice matched recordings."""
import gzip
import io
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import replay as listener
import voice_cues_willow as matched
from check_c161_command_playback_level import levels, limited, recognition
from command_level import BUILD
from run_c161_first_command_v5 import decode, decoder_api
from transition import CODEC, analyze, packets
from transition_inputs import read_noise, read_room
from voice_cues import DRY, PACK, RESULTS, binding, digest, read, save, sha, source_bank

STEM = "4090-c161-coded-command-uplink"
OUT = RESULTS / (STEM + ".json")
UART = RESULTS / (STEM + "-uart.zip")
PLAYBACK = RESULTS / (STEM + "-playback.zip")
WORK = listener.WORK / "coded-command-uplink"
PRIOR = RESULTS / (matched.STEM + "-willow.json")
FIRMWARE = RESULTS / "4090-c161-command-playback-level-p4.json"
MODES = ("original", "limited")


def sources():
    plan, old = read(matched.PLAN), read(PRIOR)
    assert sha(matched.PLAN) == old["plan_sha256"]
    bank = source_bank()
    coded = {}
    with ZipFile(PACK) as z:
        for s in json.loads(z.read("manifest.json"))["records"]:
            if s["mode"] == "coded":
                data = z.read(s["file"])
                pcm, rate = sf.read(io.BytesIO(data), dtype="int16")
                assert rate == 16000 and digest(pcm) == s["pcm_sha256"]
                coded[s["voice"]] = pcm
    for rir, kind in (("h012_", "air_conditioner"), ("h004_", "music")):
        h, room, room_hash = read_room(rir)
        noise, _, noise_hash = read_noise(kind)
        for index, (s, previous) in enumerate(zip(plan["records"], old["records"], strict=True)):
            if (s["target_id"], s["room"]) != ("coded", room):
                continue
            dry = bank[s["voice"], s["carrier"]]
            cue = coded[s["voice"]]
            assert digest(cue) == s["cue_pcm_sha256"] and previous["index"] == index
            original = listener.load_wav(s)
            assert digest(original) == previous["pcm_sha256"] == s["pcm_sha256"]
            for mode, cmd in zip(MODES, (dry, limited(dry)), strict=True):
                wet = fftconvolve(cmd.astype(float), h)
                bg = noise[(s["noise_offset"] + np.arange(len(original))) % len(noise)] * s["noise_gain"]
                begin, at = s["start"], s["end"]
                signal = bg.copy()
                signal[at:at+len(wet)] += wet
                wake = fftconvolve(cue.astype(float), h)
                signal[begin:begin+len(wake)] += wake
                q = np.rint(s["gain"]*signal)
                mic = q.clip(-32768,32767).astype("<i2")
                np.testing.assert_array_equal(mic[:at], original[:at])
                if mode == "original":
                    np.testing.assert_array_equal(mic, original)
                meta = dict(index=index, voice=s["voice"], carrier=s["carrier"], room=room, mode=mode,
                            cue_at=begin, command_at=at, mic_sha256=digest(mic), cue_sha256=digest(cue),
                            source_command=levels(cmd), room_sha256=room_hash, noise_sha256=noise_hash,
                            clipped=int(np.count_nonzero(q != mic)))
                yield meta, mic, cmd, cue, previous


def firmware():
    ref = read(FIRMWARE)
    for p, key in ((BUILD/"nyx_mac_listener.bin","listener_sha256"), (CODEC/"nyx_first_command.bin","opus_sha256"),
                   (listener.QEMU,"qemu_sha256")):
        assert sha(p) == ref[key]
    return {k:ref[k] for k in ("listener_sha256","opus_sha256","qemu_sha256")}


def boot(build, log, marker, payload, failures):
    for attempt in range(2):
        try:
            listener.boot(build, log, WORK/"flash.bin", marker, (2<<20,payload))
            return
        except (AssertionError, TimeoutError):
            if log.exists():
                with gzip.open(failures / f"{log.stem}-attempt{attempt+1}.gz","wb") as f:
                    f.write(log.read_bytes())
            if attempt:
                raise


def p4(meta, mic, detector, index, decoder, reader, previous, history, require_timely=True):
    # Each listener and encoder is cold-booted with the same signed-reference protocol.
    incoming = np.column_stack((mic, np.zeros_like(mic)))
    payload = struct.pack("<3I",0x4452594e,index,len(mic))+incoming.T.copy().tobytes()
    log = WORK / f"{index:02}-listener.log"
    boot(BUILD, log, b"LISTENER PASS", payload, WORK)
    text = log.read_text()
    details, lanes = analyze(text, dict(index=index,group="external",command_at=meta["command_at"]), incoming, detector)
    assert details["v5"] == [v["t"] for v in previous["fixed"]] and details["v5"] == details["allowed_v5"]
    assert not details["reference_v5"] and details["c161"] == previous["c161"]
    assert meta["cue_at"] <= previous["events"]["nyx"][0]
    assert details["delivered"] == (previous["events"]["nyx"][0]+255)//256*256
    selected, raw = lanes["selected"], lanes["raw"]
    assert details["samples_pure_raw"] == len(raw)
    assert digest(selected) == digest(lanes["delayed_raw"])
    capture = details["first"]
    assert capture == max(0,details["delivered"]//160*160-32000)
    if require_timely:
        assert details["delivered"] < meta["command_at"]
    tags = ("READY ","OPEN ","HITS ","C161_ALLOW ","V5_ALLOW ","CUE_ROUTE ","FINISH ","LISTENER PASS")
    history.writestr(f"{index:02}-listener.log","\n".join(l for l in text.splitlines() if l.startswith(tags))+"\n")
    details["full_listener_uart_sha256"] = sha(log)
    log.unlink()
    codec = WORK / f"{index:02}-opus.log"
    data = struct.pack("<3I",0x3552504f,index,len(raw))+selected.tobytes()+raw.tobytes()
    boot(CODEC, codec, b"UPLINK PASS", data, WORK)
    history.write(codec, f"{index:02}-opus.log")
    text = codec.read_text()
    codec.unlink()
    details["uplink"] = {}
    # This offset only uses OPEN, after the entire received capture has been decoded.
    cut = max(0, details["delivered"] - 1600 - capture)
    for lane, name in enumerate(("selected","raw")):
        encoded = packets(text,index,len(raw),lane)
        pcm = decode(decoder,encoded)
        details["uplink"][name] = dict(packets=len(encoded), payload_bytes=sum(map(len,encoded)),
            decoded_sha256=digest(pcm), crop_offset=cut, full=recognition(reader,pcm,meta["carrier"]),
            from_wake=recognition(reader,pcm[cut:],meta["carrier"]))
    return details


def write_member(archive, name, pcm):
    stream = io.BytesIO()
    sf.write(stream, pcm, 16000, format="WAV", subtype="PCM_16")
    archive.writestr(ZipInfo(name),stream.getvalue(),compress_type=ZIP_DEFLATED)
    import hashlib
    return dict(file=name, bytes=len(stream.getvalue()), samples=len(pcm),
                wav_sha256=hashlib.sha256(stream.getvalue()).hexdigest(), pcm_sha256=digest(pcm))


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    fw = firmware()
    reader, identity = listener.load_reader()
    assert identity == read(PRIOR)["asr_files"]
    decoder = decoder_api()
    rows, entries, clean_reads = [], {}, {}
    with TemporaryDirectory() as tmp, ZipFile(UART.with_suffix(".tmp"),"w",ZIP_DEFLATED) as uart, \
         ZipFile(PLAYBACK.with_suffix(".tmp"),"w",ZIP_DEFLATED) as playback:
        detector = binding(Path(tmp)/"v5.so",coded="v5")
        for index, (s, mic, cmd, cue, prev) in enumerate(sources()):
            t = prev["events"]["nyx"][0]
            delivered = (t+255)//256*256
            begin = max(0,delivered//160*160-32000)
            dry_hash = digest(cmd)
            if dry_hash not in clean_reads:
                clean_reads[dry_hash] = recognition(reader,cmd,s["carrier"])
            host = dict(clean=clean_reads[dry_hash], from_command=recognition(reader,mic[s["command_at"]:],s["carrier"]),
                        full=recognition(reader,mic[begin:],s["carrier"]),
                        from_wake=recognition(reader,mic[max(0,delivered-1600):],s["carrier"]))
            if s["mode"] == "original":
                assert host["full"] == prev["capture"]["nyx"]
            result = p4(s,mic,detector,index,decoder,reader,prev,uart)
            rows.append(dict(source=s, host=host, p4=result))
            key = (s["voice"], s["carrier"], s["mode"])
            command_at = s["command_at"]-s["cue_at"]
            wave = np.r_[cue,np.zeros(command_at-len(cue),np.int16),cmd]
            if key not in entries:
                name = f'{s["voice"]}-{s["carrier"].split()[0].lower()}-{s["mode"]}.wav'
                entries[key] = dict(voice=s["voice"],carrier=s["carrier"],mode=s["mode"],cue_sha256=digest(cue),
                                    command_at=command_at, **write_member(playback,name,wave))
            assert entries[key]["pcm_sha256"] == digest(wave)
            print(index, s["voice"], s["mode"], result["uplink"]["selected"]["full"]["complete_suffix"],flush=True)
        assert len(rows) == 32 and len(entries) == 16
        playback.writestr("manifest.json",json.dumps(dict(rate=16000, channels=1, format="PCM16",
            cue_pack_sha256=sha(PACK), command_source_sha256=sha(DRY), records=list(entries.values())),indent=2)+"\n")
        for failed in sorted(WORK.glob("*-attempt*.gz")):
            uart.write(failed,failed.name)
    UART.with_suffix(".tmp").replace(UART)
    PLAYBACK.with_suffix(".tmp").replace(PLAYBACK)
    save(OUT,dict(records=rows, asr_files=identity, decoder=decoder.opus_get_version_string().decode(), **fw,
         model_sha256=matched.MODEL_SHA, source_sha256=sha(matched.PLAN), prior_sha256=sha(PRIOR),
         code_sha256=sha(Path(__file__)), uart_sha256=sha(UART), playback_sha256=sha(PLAYBACK),
         failures=[p.name for p in sorted(WORK.glob("*-attempt*.gz"))],
         policy="Frozen four-voice matched clocks. Only the dry command changes before convolution. Silent panel reference. Keep full two-second history and compare a recognizer crop starting 100 ms before the listener delivery on the same decoded Opus packets. Primary score is the complete transcript suffix."))


if __name__ == "__main__":
    main()
