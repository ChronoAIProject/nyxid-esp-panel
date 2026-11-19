"""Compare ordinary lexical wake detection after ESP-SR and a causal reference-based echo estimate."""
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from pocketsphinx import Decoder
from scipy.signal import fftconvolve

from audit_c161_history_paths_v1 import decode, times
from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_event_transcripts_v1 import choose
from check_c161_lexical_echo import BUILD, BOT, FIRMWARE, FORWARD, OUT, QEMU, RATE, ROOMS, STEM, cases
from check_c161_midlevel_pocketsphinx import MID, STEM as MIDLEX
from check_c161_pocketsphinx import confirmations
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_stream_stretch_v1 import run_quantized
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, read, save, sha
from run_c161_first_command_v5 import transcripts
from run_c161_frontend_v5 import parse_uart


def subtract_playback(mic, ref):
    # Fit solely to the preceding three seconds. Apply each estimated FIR to the next half second.
    residual = mic.astype(np.float64)
    taps = 2048
    idx = torch.arange(taps, device="cuda")
    indices = (idx[:,None] - idx).abs()
    history, hop = 3*RATE, RATE//2
    for at in range(hop, len(mic), hop):
        begin, end = max(0, at-history), min(len(mic), at+hop)
        if not np.any(ref[begin:at]):
            continue
        x = torch.as_tensor(ref[begin:at].astype(float), device="cuda")
        y = torch.as_tensor(mic[begin:at].astype(float), device="cuda")
        size = 1 << (2*len(x)-1).bit_length()
        spec = torch.fft.rfft(x, n=size)
        auto = torch.fft.irfft(spec.abs().square(), n=size)[:taps]
        cross = torch.fft.irfft(spec.conj()*torch.fft.rfft(y, n=size), n=size)[:taps]
        matrix = auto[indices]
        matrix.diagonal().add_(0.001 * auto[0])
        filt = torch.linalg.solve(matrix, cross).cpu().numpy()
        ref_begin = max(0, at-taps+1)
        est = fftconvolve(ref[ref_begin:end].astype(float), filt)
        residual[at:end] -= est[at-ref_begin:end-ref_begin]
    rounded = np.rint(residual)
    return rounded.clip(-32768,32767).astype("<i2"), int(np.count_nonzero((rounded<-32768)|(rounded>32767)))


def infer(pcm, mid, decoder, cmn):
    x, clock = frontend(pcm)
    scales = {REF_KERNEL(model_path=str(path)).get_input_details()[0]["quantization"] for path in (mid, MODEL)}
    (scale, zero), = scales
    q = np.clip(np.rint(x / scale) + zero, -128,127).astype(np.int8)
    events = {}
    for name, path, expand in (("c161", MODEL, False), ("original", mid, False), ("expanded", mid, True)):
        scores, _ = run_quantized(q, path, expand)
        events[name] = times(decode(scores, clock))
    checked = confirmations(decoder, cmn, pcm, sorted(set(events["original"]+events["expanded"])))
    for name, proposals in (("native_proposals", events["original"]), ("both_proposals", events["original"]+events["expanded"])):
        verified = [c["confirmed"] for c in checked if c["proposal"] in proposals and c["confirmed"] is not None]
        events[name] = choose(events["c161"], verified)
    return dict(events=events, checked=checked)


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    reader, asr_identity = load_reader()
    spec = read(RESULTS / f"{MID}-training-v1.json")["reports"][0]
    mid = RESULTS / spec["path"]
    assert sha(mid) == spec["sha256"] and sha(MODEL) == MODEL_SHA
    folder = LEXICAL / "q8/files"
    model_manifest = RESULTS/"4090-c161-lexical-p4-q8-packing.json"
    manifest = read(model_manifest)
    assert all(sha(folder/name) == h for name,h in manifest["files"].items())
    decoder = Decoder(hmm=str(folder/"am"), dict=str(folder/"two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    records, scores, readings = [], {}, {}
    archive = RESULTS/f"{STEM}-uart.zip"
    temporary = OUT/"scoring-uart.zip"
    with ZipFile(temporary, "w", ZIP_DEFLATED) as uart:
        for key, mic, ref, meta in cases():
            log = OUT/f"{key}.log"
            length, chunk, prime, _, audio = parse_uart(log, ("echo",))["echo"]
            assert (length, chunk, prime) == (len(mic), 256, 81664)
            uart.write(log, key+".log")
            residual, clipped = subtract_playback(mic, ref)
            sign = meta["role"]
            external = sign in ("external_only", "external_over_reply", "simultaneous_mentions")
            self_content = sign in ("self","same_mic_external","simultaneous_mentions")
            expected = meta["external_command"] if external else "turn on the kitchen lights."
            result = dict(key=key, source=meta, uart_sha256=sha(log), residual_clipped=clipped, stages={})
            for stage, pcm, lag in (("mic",mic,0), ("p4_agc",audio["agc"],1440), ("fir",residual,0)):
                identity = digest(pcm)
                if identity not in scores:
                    scores[identity] = infer(pcm, mid, decoder, cmn)
                scored = scores[identity]
                start = meta["start"]+lag
                end = (meta["external_end"] if external else meta["self_end"]) + lag
                hits = [e for e in scored["events"]["both_proposals"] if start<=e["sample"]<end]
                row = dict(pcm_sha256=identity, target_hits=hits, evaluated_start=start, evaluated_end=end)
                if hits and (external or self_content):
                    # Inspect the existing two-second lookback on the online audio-availability clock.
                    first = max(0, (hits[0]["sample"]+159)//160*160 - 32000)
                    audio_key = digest(pcm[first:end])
                    if audio_key not in readings:
                        readings[audio_key] = transcripts(reader,pcm[first:end])
                    words = normalized(readings[audio_key]["text"])
                    want = normalized(expected)
                    row.update(first_sample=first, transcript_pcm=audio_key, exact_command_suffix=words[-len(want):]==want)
                result["stages"][stage] = row
            records.append(result)
            print(key, [len(r["target_hits"]) for r in result["stages"].values()], flush=True)
    assert len(records) == 24
    temporary.replace(archive)
    required = [FORWARD, ROOMS, FIRMWARE, model_manifest, RESULTS/f"{BOT}-v1.json", RESULTS/f"{MIDLEX}-v1.json"]
    save(RESULTS/f"{STEM}.json", dict(sources={p.name:sha(p) for p in required},
         qemu_sha256=sha(QEMU), elf_sha256=sha(BUILD/"nyx_first_command.elf"), uart_archive_sha256=sha(archive),
         model_sha256=dict(c161=MODEL_SHA, midlevel=spec["sha256"]), asr=asr_identity,
         device=torch.cuda.get_device_name(), fir_taps=2048, fit_history=48000, fit_hop=8000, ridge_ratio=.001,
         records=records,scores=scores,transcripts=readings))


if __name__ == "__main__":
    main()
