"""Check the upstream residual echo postfilter on the same ordinary requests, including misses."""
import re
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from pocketsphinx import Decoder
from scipy.signal import correlate, correlation_lags

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_lexical_echo import OUT, QEMU, cases
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_speex_p4 import ARCHIVE, EXPECTED, parse
from run_c161_first_command_v5 import boot, transcripts
from score_c161_lexical_echo import infer

STEM = "4090-c161-speex-residual"
BASE = RESULTS / "4090-c161-speex-p4-float.json"
LAG = 160   # overlap-add buffer, N3=2*N-frame_size in upstream preprocess.c


def speech_reading(reader, pcm, first, end, want):
    result = transcripts(reader, pcm[first:end])
    words = normalized(want)
    return dict(first=first, end=end, transcript=result, exact_suffix=normalized(result["text"])[-len(words):] == words)


def main():
    output = RESULTS / f"{STEM}.json"
    archive = RESULTS / f"{STEM}-uart.zip"
    if archive.exists() or output.exists():
        raise FileExistsError(STEM)
    before = read(BASE)
    original = {r["key"]: r for r in before["records"]}
    reader, asr = load_reader()
    assert asr == before["asr"] and sha(QEMU) == before["qemu_sha256"] and sha(ARCHIVE) == EXPECTED
    mid_spec = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"][0]
    mid = RESULTS / mid_spec["path"]
    manifest = RESULTS / "4090-c161-lexical-p4-q8-packing.json"
    files = LEXICAL / "q8/files"
    assert sha(mid) == mid_spec["sha256"] and sha(MODEL) == MODEL_SHA
    assert all(sha(files / n) == h for n, h in read(manifest)["files"].items())
    decoder = Decoder(hmm=str(files / "am"), dict=str(files / "two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    root = OUT / "residual"
    root.mkdir(exist_ok=True)
    build = OUT / "build-post"
    rows = []
    with ZipFile(root / "uart.zip", "w", ZIP_DEFLATED) as uart:
        for key, mic, ref, meta in cases():
            baseline = original.pop(key)
            assert baseline["source"] == meta
            payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
            log = root / f"{key}.log"
            boot(build, log, root / "flash.bin", b"SPEEX PASS", (2 << 20, payload))
            pcm, costs, heap = parse(log, len(mic))
            settings = re.findall(r"SPEEX_POST noise=(-?\d+) echo=(-?\d+) active=(-?\d+) agc=(\d+)", log.read_text())
            assert len(settings) == 1
            scored = infer(pcm, mid, decoder, cmn)
            external = meta["role"] in ("external_only", "external_over_reply", "simultaneous_mentions")
            end = (meta["external_end"] if external else meta["self_end"]) + LAG + 160
            start = meta["start"] + LAG
            hits = [e for e in scored["events"]["both_proposals"] if start <= e["sample"] < end]
            row = dict(key=key, source=meta, detections=scored, hits=hits, baseline_hits=baseline["hits"],
                       baseline_exact_suffix=baseline.get("exact_suffix"), pcm_sha256=digest(pcm), settings=settings[0],
                       heap=heap, uart_sha256=sha(log), median_us=float(np.median(costs)), max_us=max(costs),
                       over_10ms=sum(v > 10000 for v in costs))
            if meta["role"] == "external_only":
                corr = correlate(pcm.astype(float), mic.astype(float), method="fft")
                lags = correlation_lags(len(pcm), len(mic))
                row["measured_direct_lag"] = int(lags[np.argmax(corr)])
            if hits and meta["role"] != "reply_only":
                first = max(0, (hits[0]["sample"] + 159) // 160 * 160 - 32000)
                want = meta["external_command"] if external else "turn on the kitchen lights."
                row["capture"] = speech_reading(reader, pcm, first, end, want)
            if meta["role"] == "external_over_reply":
                # Diagnostic transcription uses the known offline span, including when KWS never woke.
                row["known_span_only"] = speech_reading(reader, pcm, start-3200, end, meta["external_command"])
            rows.append(row)
            uart.write(log, log.name)
            print(key, len(hits), row.get("capture", {}).get("exact_suffix"),
                  row.get("known_span_only", {}).get("exact_suffix"), flush=True)
    assert not original
    (root / "uart.zip").replace(archive)
    save(output, dict(baseline_sha256=sha(BASE), archive_sha256=sha(ARCHIVE), qemu_sha256=sha(QEMU),
                      elf_sha256=sha(build / "nyx_speex_echo.elf"), app_sha256=sha(build / "nyx_speex_echo.bin"),
                      asr=asr, c161_sha256=MODEL_SHA, midlevel_sha256=sha(mid), q8_manifest_sha256=sha(manifest),
                      expected_lag_samples=LAG, uart_archive_sha256=sha(archive), records=rows))


if __name__ == "__main__":
    main()
