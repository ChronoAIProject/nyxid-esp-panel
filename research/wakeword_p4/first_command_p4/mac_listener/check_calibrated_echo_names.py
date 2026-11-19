"""Challenge calibration with target, approximately spoken target, competing names and connected commands."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import fftconvolve
import torch

import fit_spoken_echo_path as fit
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import transcripts
from check_c161_v5_far_command_noise import normalized

spoken, ch = fit.spoken, fit.ch
OUT = ch.RESULTS / "4090-c161-calibrated-echo-names.json"
COEFF = OUT.with_suffix(".npz")
HOP = 256


def estimate(ref, h):
    # Uniform partitioned overlap-add. At block b, only spectra 0..b have arrived.
    parts = len(h)//HOP
    assert len(h) == parts*HOP
    kernels = rfft(h.astype(np.float32).reshape(parts,HOP), n=2*HOP, axis=1)
    blocks = np.pad(ref.astype(np.float32),(0,-len(ref)%HOP)).reshape(-1,HOP)
    prior = np.zeros_like(kernels)
    tail = np.zeros(HOP, np.float32)
    output = np.empty_like(blocks)
    for b, block in enumerate(blocks):
        prior[1:] = prior[:-1].copy()
        prior[0] = rfft(block, n=2*HOP)
        convolved = irfft((kernels*prior).sum(axis=0), n=2*HOP)
        output[b] = convolved[:HOP] + tail
        tail = convolved[HOP:]
    return output.ravel()[:len(ref)]


def clocks(trace, independent, cutoff, char_us, five_us):
    merge = spoken.natural.base.scoring.merged
    raw_c161 = [(t+255)//256*256 for t in independent]
    cleaned_c161 = [(t+255)//256*256 for t in trace["c161"] if t >= cutoff]
    c161 = sorted(set(raw_c161 + cleaned_c161))
    simple, serial = [], []
    done = 0
    for t in trace["live_ends"]:
        if t < cutoff:
            continue
        at = (t+255)//256*256
        check = trace["checks"][t]
        cost = char_us + (five_us if check["accepted"] else 0)
        done = max(at,done) + (16*cost+999)//1000
        if check["pass"]:
            simple.append(at)
            serial.append(done)
    return dict(available=merge(c161,simple), estimated=merge(c161,serial))


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    net = spoken.natural.Detector()
    previous = ch.read(fit.OUT)
    assert net.identity == previous["models"]
    previous_pairs = {(r["mic_sha256"],r["ref_sha256"]):r for r in previous["records"]}
    plan = ch.read(spoken.connected.PLAN)
    sources = {}
    for s in plan["records"]:
        pcm = spoken.natural.replay.probe.dialogue.piper.load_wav(s)
        assert ch.digest(pcm) == s["pcm_sha256"]
        sources[s["index"]] = pcm
    char_file = ch.RESULTS/"4090-c161-early-readout-p4-character.json"
    five_file = ch.RESULTS/"4090-c161-early-readout-p4-five.json"
    char_us = int(np.ceil(np.percentile([r["invoke_us"]+r["decode_us"] for r in ch.read(char_file)["records"]],95)))
    five_us = int(np.ceil(np.percentile([r["invoke_us"] for r in ch.read(five_file)["records"]],95)))
    reader, identity = ch.load_reader()
    assert identity == previous["asr"]
    rows, filters, filter_records, readings = [], {}, {}, {}
    with TemporaryDirectory() as tmp:
        coded = ch.binding(Path(tmp)/"v5.so", coded="v5")
        for meta, self_mic, ref in spoken.extra_cases():
            if meta["mode"] != "self_only":
                continue
            bot = meta["bot"]
            if bot not in filters:
                h = fit.fit(self_mic,ref)
                actual = previous_pairs[ch.digest(self_mic),ch.digest(ref)]["fit"]
                assert ch.digest(h) == actual["filter_sha256"]
                # Coefficients are portable little-endian binary32. Validate their streaming output.
                filters[bot] = h.astype("<f4")
                predicted = estimate(ref, filters[bot])
                double = fftconvolve(ref.astype(float),h)[:len(ref)]
                max_diff = float(np.max(abs(predicted-double)))
                assert max_diff < 1, max_diff
                valid = spoken.explained(self_mic[fit.FIT_END:fit.READY].astype(float),predicted[fit.FIT_END:fit.READY])
                assert valid["rms"]>=64 and valid["explained"]>=.80
                filter_records[bot] = dict(max_reference_prediction_diff_pcm=max_diff, validation=valid,
                                           float32_sha256=ch.digest(filters[bot]), original_fit=actual)
            prediction = estimate(ref, filters[bot])
            room, name, digest = ch.read_room(Path(meta["room"]).name[:5])
            assert name == meta["room"] and digest == meta["rir_sha256"]
            echo = fftconvolve(ref.astype(float),room)[:len(ref)] * meta["mic_echo_gain"]
            assert np.array_equal(np.rint(echo).astype("<i2"),self_mic)
            start = meta["cue_at"]
            cutoff = (fit.READY+HOP-1)//HOP*HOP
            for s in (s for s in plan["records"] if s["voice"] == meta["voice"]):
                clip = sources[s["index"]][s["start"]:]
                length = s["end"]-s["start"]
                end = start+len(clip)-16000
                cmd = start+length
                for db in (-6,0,6):
                    gain = 10**(db/20)*np.linalg.norm(echo[start:cmd])/np.linalg.norm(clip[:length].astype(float))
                    external = np.zeros(len(ref))
                    external[start:start+len(clip)] = gain*clip
                    mic = np.rint(external+echo).clip(-32768,32767).astype("<i2")
                    unaccompanied = np.rint(external).clip(-32768,32767).astype("<i2")
                    clean = np.rint(mic.astype(float)-prediction).clip(-32768,32767).astype("<i2")
                    clean[:cutoff] = 0
                    key = (ch.digest(mic),ch.digest(ref))
                    if s["target_id"] in ("fixed-v5","Hey Nyx") and s["carrier"] == spoken.connected.CARRIERS[0]:
                        assert key in previous_pairs
                    original = net.score(mic)
                    fixed = [e["t"] for e in detect(coded,[mic])]
                    row = dict(voice=s["voice"], role=s["target_id"], command=s["carrier"], db=db, index=s["index"],
                               bot=bot, start=start, command_at=cmd, end=end, mic_sha256=key[0], ref_sha256=key[1],
                               raw_panel=spoken.natural.base.scoring.merged(original["c161"], fixed), stages={})
                    for name,pcm,begin,independent in (("raw",mic,0,original["c161"]),
                            ("external_only",unaccompanied,0,[]), ("filtered",clean,cutoff,original["c161"])):
                        trace = original if name=="raw" else net.score(pcm)
                        schedule = clocks(trace,independent,begin,char_us,five_us)
                        stage = dict(pcm_sha256=ch.digest(pcm), c161=trace["c161"], proposals=trace["proposals"],
                                     events=trace["events"], **schedule)
                        hits = [t for t in schedule["estimated"] if start<=t<end]
                        stage["before_command"] = any(t<cmd for t in hits)
                        if hits and s["target_id"] in ("Hey Nyx","Hey Nicks","fixed-v5"):
                            first = max(0,((hits[0]+159)//160)*160-32000)
                            crop = pcm[first:end]
                            digest = ch.digest(crop)
                            if digest not in readings:
                                readings[digest] = transcripts(reader,crop)
                            words, want = normalized(readings[digest]["text"]), normalized(s["carrier"])
                            stage.update(first=first, capture_sha256=digest, complete=words[-len(want):]==want)
                        row["stages"][name] = stage
                    rows.append(row)
            print(meta["voice"],len(rows),flush=True)
    with COEFF.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out, **filters)
    COEFF.with_suffix(".tmp").replace(COEFF)
    assert len(rows)==180 and len(filters)==2
    ch.save(OUT,dict(records=rows, texts=readings, models=net.identity, asr=identity, filters=filter_records,
        coeff_sha256=ch.sha(COEFF), block=HOP, char_us=char_us, five_us=five_us, cutoff_samples=cutoff,
        hashes={p.name:ch.sha(p) for p in (fit.OUT,spoken.connected.PLAN,char_file,five_file,Path(__file__))},
        policy="Use all four voices, five address forms, three command carriers and three levels at the same "
        "two frozen static playback paths. Test room-specific float32 FIR in 256-sample blocks after validating "
        "a known panel-only calibration. Zero prefix on the residual branch, separate raw C161. Estimated adds "
        "per-verifier QEMU P95 service times on one serial queue; actual silicon and FIR compute not timed. "
        "Transcript reads the two-second stage-PCM lookback without Opus. No independent holdout."))


if __name__ == "__main__":
    main()
