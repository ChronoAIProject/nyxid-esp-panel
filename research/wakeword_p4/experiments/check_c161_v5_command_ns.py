"""Compare native P4 ESP-SR noise suppression on the same 36 frozen v5 command streams."""
import argparse
import hashlib
import struct
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import correlate
import soundfile as sf

from check_c161_coded_cue_v4 import command_streams
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from run_c161_first_command_v5 import QEMU, WORK, boot, transcripts
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-v5-command-ns"
BUILD = WORK / "ns-build"
OUT = WORK / "ns-records"
REFERENCE = RESULTS / "4090-c161-coded-cue-v5-commands.json"
FAR = RESULTS / "4090-c161-v5-far-command-noise.json"


def digest(pcm):
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def cases():
    reference = read(REFERENCE)
    expected = {(s["source"]["voice"], s["source"]["scene"], s["source"]["channel"]): s
                for s in reference["records"] if s["source"]["role"] == "v5"}
    cue = load_wav(dict(path=WAV, sha256=reference["cue_sha256"]))
    rows = []
    for source, pcm in command_streams(cue):
        # The shared renderer uses the v4 role label for the waveform supplied by its caller.
        if source["role"] != "v4":
            continue
        key = (source["voice"], source["scene"], source["channel"])
        ref = expected.pop(key)
        assert digest(pcm) == ref["pcm_sha256"] and source["end"] == ref["source"]["end"]
        rows.append((ref["source"], pcm))
    assert len(rows) == 36 and not expected
    return rows


def replay(rows):
    OUT.mkdir(exist_ok=True)
    for group in range(0, len(rows), 3):
        payload = bytearray(struct.pack("<I", 3))
        for i in range(group, group+3):
            src, pcm = rows[i]
            start = src["start"] - 2048
            payload.extend(struct.pack("<4I", i, len(pcm), start, 1))
            payload.extend(pcm.astype("<i2", copy=False).tobytes())
        boot(BUILD, OUT / f"{group}.log", OUT / "flash.bin", b"NS PASS", (2 << 20, payload))


def frames(log, rows, group):
    current = None
    complete = []
    for line in log.read_text().splitlines():
        if line.startswith("BEGIN "):
            _, idx, length, start, mode, allocated = line.split()
            current = int(idx)
            assert current == group + len(complete)
            src, pcm = rows[current]
            assert (int(length), int(start), int(mode)) == (len(pcm), src["start"]-2048, 1)
            origin = int(start) // 160 * 160
            raw = bytearray()
        elif line.startswith("PCM "):
            _, at, data = line.split()
            assert current is not None and int(at) == origin + len(raw)//2
            chunk = bytes.fromhex(data)
            assert len(chunk) == 320
            raw.extend(chunk)
        elif line.startswith("END "):
            assert int(line.split()[1]) == current
            audio = np.frombuffer(raw, dtype="<i2")
            assert origin+len(audio) == ((len(pcm)+1600+159)//160)*160
            complete.append((origin, audio, int(allocated)))
            current = None
    assert "NS PASS" in log.read_text() and current is None and len(complete) == 3
    return complete


def aligned_lag(source, output, origin, onset):
    segment = source[onset:onset+16000].astype(np.float64)
    observed = output[onset-origin:onset-origin+len(segment)+1024].astype(np.float64)
    energy = np.r_[0, np.cumsum(observed**2)]
    norm = np.sqrt(np.maximum(1, (energy[len(segment):]-energy[:-len(segment)]) * (segment @ segment)))
    corr = correlate(observed, segment, "valid", method="fft") / norm
    lag = int(np.argmax(corr))
    assert lag < 1024
    return lag, float(corr[lag])


def score(rows):
    from faster_whisper import WhisperModel
    model = Path(".local/wakeword/whisper-small-en")
    identity = read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")["files"]
    model_hashes = {str(p): sha(p) for p in model.iterdir() if p.is_file()}
    assert model_hashes == {k: v for k, v in identity.items() if Path(k).parent == model}
    reader = WhisperModel(str(model), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    far = {(r["voice"], r["scene"]): r for r in read(FAR)["records"] if r["noise_reduction_db"] == 0}
    records = []
    for group in range(0, len(rows), 3):
        for j, (origin, filtered, allocated) in enumerate(frames(OUT/f"{group}.log", rows, group)):
            src, pcm = rows[group+j]
            onset = src["end"]
            lag, similarity = aligned_lag(pcm, filtered, origin, onset)
            processed = filtered[onset-origin+lag:len(pcm)-origin+lag]
            command = pcm[onset:]
            assert len(processed) == len(command)
            ref = far[src["voice"], src["scene"]]
            expected = normalized(ref["expected"])
            row = dict(voice=src["voice"], scene=src["scene"], channel=src["channel"], expected=ref["expected"],
                       full_input_sha256=digest(pcm), command_start=onset, allocated_bytes=allocated,
                       output_origin=origin, returned_pcm_sha256=digest(filtered), lag=lag, correlation=similarity)
            for label, audio in (("input", command), ("ns", processed)):
                result = transcripts(reader, audio)
                words = normalized(result["text"])
                row[label] = dict(transcript=result, first=words[:1] == expected[:1], exact=words == expected,
                                  command_sha256=digest(audio))
            if src["channel"] == "far":
                assert row["input"]["transcript"]["text"] == ref["transcript"]["text"] and digest(pcm) == ref["pcm_sha256"]
            records.append(row)
    archive = RESULTS / f"{STEM}-uart.zip"
    with ZipFile(archive, "x", ZIP_DEFLATED) as z:
        for group in range(0, len(rows), 3):
            z.write(OUT/f"{group}.log", f"{group}.log")
    # The prebuilt library identity was verified against the root dependency lock and registry checksums.
    prior = read(RESULTS/"4090-c161-first-command-frontend-v5.json")["dependencies"]["espressif/esp-sr"]
    sr = Path(next(s.split("=",1)[1] for s in (BUILD/"CMakeCache.txt").read_text().splitlines() if s.startswith("NYX_SR_COMPONENT:")))
    assert (sr/".component_hash").read_text().strip() == prior["component_hash"]
    assert sha(sr/"lib/esp32p4/libesp_audio_processor.a") == prior["files"]["lib/esp32p4/libesp_audio_processor.a"]
    save(RESULTS/f"{STEM}.json", dict(version=prior["version"], sr_component_hash=prior["component_hash"],
         processor_sha256=prior["files"]["lib/esp32p4/libesp_audio_processor.a"], model_files=model_hashes,
         qemu_sha256=sha(QEMU), elf_sha256=sha(BUILD/"nyx_first_command.elf"), uart_sha256=sha(archive),
         source_report_sha256=sha(REFERENCE), far_report_sha256=sha(FAR),
         settings=dict(frame_ms=10, mode=1, rate=16000, input="continuous mono from sample zero",
                       route="separate ASR capture test, detector input not changed"), records=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", nargs="+", choices=["replay", "score"])
    data = cases()
    for action in parser.parse_args().actions:
        {"replay": replay, "score": score}[action](data)
