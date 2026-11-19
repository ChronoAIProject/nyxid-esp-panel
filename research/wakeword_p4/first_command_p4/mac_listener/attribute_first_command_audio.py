"""Trace the frozen v5 command through source, listener FIFO, Opus and two unprompted GPU readers."""
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import correlate
import soundfile as sf
from faster_whisper import WhisperModel

import coded_fixed_transport as rooms
import connected_nick_willow as connected
import fixed_cue_dialogue_p4 as piper
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

ch, uplink = rooms.ch, piper.uplink
TURBO = Path(".local/wakeword/whisper-large-v3-turbo-ct2")
REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
MODEL_HASH = "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"
STEM = "4090-c161-first-command-audio-attribution"


def wav(pack, entry, name, wav_digest, pcm_digest):
    data = pack.read(entry[name])
    assert sha256(data).hexdigest() == entry[wav_digest]
    signal, rate = sf.read(io.BytesIO(data), dtype="int16")
    assert rate == 16000 and signal.ndim == 1 and ch.digest(signal) == entry[pcm_digest]
    return signal


def piper_inputs(report):
    original = read(piper.host.study.dialogue.corpus.PLAN)["records"]
    with ZipFile(RESULTS/(piper.STEM+"-playback.zip")) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert manifest["report_sha256"] == sha(RESULTS/(piper.STEM+".json"))
        for row, entry in zip(report["records"], manifest["records"], strict=True):
            mic = wav(pack, entry, "file", "wav_sha256", "pcm_sha256")
            src, meta = original[row["stream"]], row["source"]
            span = src["timeline"][row["span"]]
            raw = piper.host.piper.load_wav(src)
            start = meta["command_at"] + row["stream_start"]
            dry = raw[start:span["end"]]
            np.testing.assert_array_equal(dry, mic[meta["command_at"]:meta["command_at"]+len(dry)])
            assert not mic[meta["command_at"]+len(dry):].any()
            yield meta, mic, dry, raw[span["start"]:span["end"]], row["board"]


def kokoro_inputs(report):
    plan = read(connected.PLAN)
    assert sha(connected.PACK) == plan["audio_sha256"] and sha(connected.PLAN) == report["plan_sha256"]
    with ZipFile(connected.PACK) as pack:
        for row in report["records"]:
            if row["role"] != "fixed-v5":
                continue
            idx = row["index"]
            src, paired = plan["records"][idx], plan["records"][idx-1]
            assert paired["target_id"] == "Hey Nyx" and src["suffix_sha256"] == paired["suffix_sha256"]
            mic, original = [wav(pack, s, "member", "sha256", "pcm_sha256") for s in (src, paired)]
            at = src["end"]
            assert not mic[-16000:].any()
            np.testing.assert_array_equal(mic[at:], original[at:])
            meta = dict(voice=src["voice"], carrier=src["carrier"], command_at=at, cue_at=src["start"])
            yield meta, mic, original[at:-16000], original[32000:-16000], row["board"]


def room_inputs(report):
    materials = (x for x in rooms.materials() if x[2] == "advance")
    for (_, s, _, mic, cue, cmd), row in zip(materials, report["records"], strict=True):
        assert ch.digest(mic) == row["mic_sha256"] and ch.digest(cmd) == s["command_pcm_sha256"]
        meta = {k:row[k] for k in ("voice","carrier","room","cue_at","command_at")}
        assert len(cue) == meta["command_at"]-meta["cue_at"]
        yield meta, mic, cmd, None, row["external"]


def codec_clock(source, decoded, at):
    # Estimate decoded lag on the first second of the command, allowing up to 40 ms.
    x = source[at:at+16000].astype(float)
    y = decoded[at:at+len(x)+640].astype(float)
    assert len(y) == len(x)+640
    dots = correlate(y, x, mode="valid", method="fft")
    power = np.r_[0., np.cumsum(y*y)]
    corr = dots / np.sqrt(np.maximum(1., (x@x)*(power[len(x):]-power[:-len(x)])))
    lag = int(corr.argmax())
    assert 0 < lag < 640
    return dict(lag=lag, correlation=float(corr[lag]))


def compare(meta, mic, dry, original, b, uart, decoder, readers):
    i, at, start, end = b["index"], meta["command_at"], b["first"], b["emitted"]
    assert meta["cue_at"] < b["delivered"] < at and b["samples_pure_raw"] == b["count"] == end-start
    mic_padded = np.pad(mic, (0, max(0, end-len(mic))))
    text = uart.read(f"{i:02}-opus.log").decode().replace("\r\n", "\n")
    assert "UPLINK PASS" in text
    versions = {}
    diagnostic = {}
    got = {}
    for lane, (name, delay) in enumerate((("selected",1440), ("raw",0))):
        fifo = np.pad(mic_padded,(delay,0))[start:end]
        assert ch.digest(fifo) == b["capture_sha256"][name]
        old = b["uplink"][name]
        packets = uplink.packets(text, i, len(fifo), lane)
        decoded = uplink.decode(decoder, packets)
        assert len(packets) == old["packets"] and ch.digest(decoded) == old["decoded_sha256"]
        enc_start = at + delay - start
        clock = codec_clock(fifo, decoded, enc_start)
        dec_start = enc_start + clock["lag"]
        # Use the entire labelled dry command duration on all three representations.
        # Both silent-tail corpora keep the same source interval across both lanes.
        n = len(dry)
        assert 0 <= enc_start < enc_start+n <= len(fifo) and dec_start+n <= len(decoded)
        np.testing.assert_array_equal(fifo[enc_start:enc_start+n], mic_padded[at:at+n])
        cut = b["delivered"] - 1600 - start
        assert cut == old["crop_offset"] and cut >= 0
        versions[name+"_opus_command"] = decoded[dec_start:dec_start+n]
        diagnostic[name] = dict(**clock, capture_sha256=ch.digest(fifo), decoded_sha256=ch.digest(decoded),
                                command_offset=dec_start, packets=len(packets), before_command_samples=enc_start,
                                after_command_samples=len(fifo)-enc_start-n)
        if name == "selected":
            versions["selected_open_minus_100ms"] = decoded[cut:]
            versions["selected_full"] = decoded
            got = {"selected_full":old["full"], "selected_open_minus_100ms":old["from_wake"]}
    versions.update(source_command=dry, mic_fifo_command=mic[at:at+len(dry)])
    if original is not None:
        versions["original_address_and_command"] = original
    cache, scores = {}, {}
    for stage, pcm in versions.items():
        digest = ch.digest(pcm)
        if digest not in cache:
            cache[digest] = {k:uplink.recognition(reader, pcm, meta["carrier"]) for k,reader in readers.items()}
        pair = cache[digest]
        scores[stage] = dict(pcm_sha256=digest, samples=len(pcm),
                             texts={k:{f:r[f] for f in ("text","exact","complete_suffix")} for k,r in pair.items()})
        if stage in got:
            assert pair["small.en"]["text"] == got[stage]["text"]
    return dict(index=i, source=meta, mic_sha256=ch.digest(mic), opened=b["delivered"],
                capture_start=start, clocks=diagnostic, stages=scores)


def main():
    turbo_files = {f:sha(TURBO/f) for f in ("config.json","preprocessor_config.json","tokenizer.json",
                                           "vocabulary.json","model.bin")}
    assert turbo_files["model.bin"] == MODEL_HASH
    small, identity = ch.load_reader()
    turbo = WhisperModel(str(TURBO), device="cuda", compute_type="float16", num_workers=1, cpu_threads=1)
    assert small.model.device == turbo.model.device == "cuda"
    readers = {"small.en":small, "large-v3-turbo":turbo}
    decoder = uplink.decoder_api()
    rows, archives = [], {}
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict(exact=0,complete_suffix=0,n=0))))
    groups = (("piper",piper.STEM,piper.STEM,piper_inputs),
              ("kokoro",connected.STEM,connected.STEM,kokoro_inputs),
              ("rooms",rooms.fixed.STEM+"-p4",rooms.fixed.STEM,room_inputs))
    for group, report_stem, uart_stem, inputs in groups:
        path = RESULTS/(report_stem+".json")
        report = read(path)
        uart_path = RESULTS/(uart_stem+"-uart.zip")
        assert sha(uart_path) == report.get("uart_sha256",report.get("archive_sha256"))
        assert identity == report.get("asr", report.get("asr_files"))
        archives[group] = dict(report_sha256=sha(path), uart_sha256=sha(uart_path))
        with ZipFile(uart_path) as uart:
            assert uart.testzip() is None
            for meta,mic,dry,original,board in inputs(report):
                result = compare(meta,mic,dry,original,board,uart,decoder,readers)
                rows.append(dict(group=group,**result))
                for stage, item in result["stages"].items():
                    for name,score in item["texts"].items():
                        count = counts[group][stage][name]
                        count["n"] += 1
                        for flag in ("exact","complete_suffix"):
                            count[flag] += score[flag]
                print(group,board["index"],flush=True)
    assert len(rows) == 46
    save(RESULTS/(STEM+".json"),dict(records=rows, summary=counts, sources=archives, small_en=identity,
        turbo=dict(repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",revision=REVISION,files=turbo_files),
        software={k:version(k) for k in ("faster-whisper","ctranslate2","scipy")}, script_sha256=sha(Path(__file__)),
        cue_sha256=sha(rooms.fixed.CUE), policy="Same fixed handoff cue on 18 Piper and 12 Kokoro direct mic "
        "recordings and 16 two-room inputs. No holdout or human speech. Zero playback reference. Check both P4 "
        "FIFO hashes against raw mic and its delayed copy. For diagnosis crop the same labelled source interval "
        "before and after Opus using measured lane correlation delay. The deployable OPEN-minus-100-ms window "
        "uses only the recorded trigger. Full history and that window use the entire decoded tail. Same unprompted "
        "English beam-3 FP16 GPU decoding, word timestamps, temperature zero, no VAD. Words scored after the "
        "fixed normalization. Original utterance includes its original spoken address for the direct sources."))


if __name__ == "__main__":
    main()
