"""Fit a stationary playback path from a panel-only prefix and test the frozen wake detector."""
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import torch

import check_spoken_playback_echo as spoken
import replay_spoken_echo_frontends as front
from check_c161_v5_far_command_noise import normalized
from run_c161_first_command_v5 import transcripts

ch = spoken.ch
OUT = ch.RESULTS / "4090-c161-calibrated-spoken-echo.json"
RATE = 16000
TAPS = 4096
FIT_END = 4 * RATE
READY = FIT_END + 12000
RIDGE = 1e-6


def fit(mic, ref):
    # Each row contains the current reference and its preceding taps of history.
    x = torch.from_numpy(ref[:FIT_END].astype(np.float64)).cuda() / 32768
    y = torch.from_numpy(mic[:FIT_END].astype(np.float64)).cuda() / 32768
    windows = torch.nn.functional.pad(x, (TAPS-1, 0)).unfold(0, TAPS, 1)
    gram = torch.zeros((TAPS, TAPS), device="cuda", dtype=x.dtype)
    xy = torch.zeros(TAPS, device="cuda", dtype=x.dtype)
    for at in range(0, FIT_END, TAPS):
        block = windows[at:at+TAPS].contiguous()
        gram.addmm_(block.T, block)
        xy.addmv_(block.T, y[at:at+TAPS])
    scale = float(gram.diagonal().mean())
    assert scale > 0
    gram.diagonal().add_(scale*RIDGE)
    # The oldest reference sample is the first column of the regression matrix.
    return torch.cholesky_solve(xy[:,None], torch.linalg.cholesky(gram)).ravel().flip(0).cpu().numpy()


def residual(mic, ref, impulse):
    estimate = fftconvolve(ref.astype(float), impulse)[:len(ref)]
    valid = spoken.explained(mic[FIT_END:READY].astype(float), estimate[FIT_END:READY])
    qualified = valid["rms"] >= 64 and valid["explained"] >= .80
    q = np.rint(mic.astype(float)-estimate)
    return q.clip(-32768,32767).astype("<i2"), dict(calibration=valid, qualified=qualified,
        clipped=int(np.count_nonzero((q[READY:] < -32768) | (q[READY:] > 32767))))


def main():
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    net = spoken.natural.Detector()
    old = ch.read(front.OUT)
    assert old["models"] == net.identity
    reader, identity = ch.load_reader()
    assert identity == old["asr"]
    filters, scores, texts = {}, {}, {}
    rows = []
    def scored(pcm):
        key = ch.digest(pcm)
        if key not in scores:
            trace=net.score(pcm)
            scores[key] = {k:trace[k] for k in ("c161", "proposals", "events")}
        return key
    # Match the external-only PCM by source, gain and cue identity.
    near = {}
    for meta, mic, _ in spoken.extra_cases():
        if meta["mode"] == "external_only":
            near[meta["original_pcm_sha256"], meta["external_gain"]] = mic
    by_hash = {(r["mic_sha256"], r["ref_sha256"]):r for r in old["records"] if r["group"]=="additional_pairs"}
    for meta, mic, ref in spoken.extra_cases():
        parent = by_hash[ch.digest(mic), ch.digest(ref)]
        start, end, cmd = parent["start"], parent["end"], parent["command_at"]
        raw = scored(mic)
        raw_events = scores[raw]["events"]
        assert raw_events == old["signals"][raw]["events"]
        row = dict(source=meta, mic_sha256=raw, ref_sha256=ch.digest(ref), start=start, end=end, command_at=cmd, runs={})
        signals = {"raw":mic}
        if ref.any():
            pair = ch.digest(mic[:FIT_END]), ch.digest(ref[:FIT_END])
            if pair not in filters:
                filters[pair] = fit(mic,ref)
            h = filters[pair]
            clean, info = residual(mic,ref,h)
            row["fit"] = dict(mic_prefix=pair[0], ref_prefix=pair[1], filter_sha256=ch.digest(h), **info)
            if info["qualified"]:
                dry = near[meta["original_pcm_sha256"],meta["external_gain"]] if cmd else np.zeros_like(mic)
                shifted = ref.copy()
                # Keep alignment throughout validation, then delay only the reported electrical reference by 16 ms.
                shifted[start:] = ref[start-256:-256]
                shifted_clean, _ = residual(mic,shifted,h)
                for name, tail, prefix in (("warm_filter", clean, mic), ("cold_filter", clean, None),
                                           ("cold_raw", mic, None), ("warm_oracle", dry, mic),
                                           ("cold_oracle", dry, None), ("cold_shifted", shifted_clean, None)):
                    passed = tail.copy()
                    passed[:READY] = prefix[:READY] if prefix is not None else 0
                    signals[name] = passed
                if cmd:
                    for name, pcm in (("raw", mic), ("filter", clean), ("shifted", shifted_clean)):
                        error = pcm[start:cmd].astype(float) - dry[start:cmd]
                        target = dry[start:cmd].astype(float)
                        row.setdefault("cue_snr_db", {})[name] = float(10*np.log10(max(1.,target@target)/max(1.,error@error)))
        for name, pcm in signals.items():
            k = scored(pcm)
            events = scores[k]["events"]
            # Preserve the raw independent C161 route. The new residual branch becomes available at READY.
            merged = raw_events if name == "raw" else spoken.natural.base.scoring.merged(
                scores[raw]["c161"], [t for t in events if t>=READY])
            hits = [t for t in merged if start <= t < end]
            run = dict(pcm_sha256=k, events=merged, hits=hits, early=any(t<cmd for t in hits) if cmd else None)
            if cmd and hits and name in ("raw","cold_filter","warm_filter","cold_shifted"):
                first = max(0, ((hits[0]+159)//160)*160-32000)
                crop = pcm[first:end]
                key = ch.digest(crop)
                if key not in texts:
                    texts[key] = transcripts(reader,crop)
                words, want = normalized(texts[key]["text"]), normalized(meta["carrier"])
                run.update(capture_start=first, capture_sha256=key, complete=words[-len(want):]==want)
            row["runs"][name] = run
        rows.append(row)
    assert len(rows)==52
    ch.save(OUT,dict(records=rows, signals=scores, texts=texts, models=net.identity, asr=identity,
                     method=dict(device=device, fit_samples=FIT_END, ready_samples=READY, taps=TAPS,
                                 ridge_diagonal_ratio=RIDGE, filters=len(filters)),
                     hashes={p.name:ch.sha(p) for p in (front.OUT, spoken.OUT, Path(__file__))},
                     policy="Fit causal FIR on the first 4 s of known panel-only audio using double precision "
                     "CUDA ridge regression; validate on the next 750 ms at RMS >=64 and explained energy >=80%. "
                     "Only after validation enable the residual wake route. Cold prefixes are silence, warm prefixes "
                     "are the original microphone. Oracle uses the paired clean external track only as a diagnostic. "
                     "All runs retain raw C161. Notification and two-second full-suffix ASR use sample clocks "
                     "without DSP/verification execution time or Opus. Static synthetic path and synced reference."))


if __name__ == "__main__":
    main()
