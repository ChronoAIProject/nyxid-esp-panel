"""Freeze quiet/playback pairs on four other voices, replay both P4 filters and score the candidates."""
import hashlib
import re
import struct
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from pocketsphinx import Decoder
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_event_transcripts_v1 import choose
from check_c161_lexical_echo import BOT, OUT
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_hold_rooms import RATE, LENGTH, rir
from check_c161_speex_residual import BASE, LAG, STEM as POST, speech_reading
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import QEMU, boot
from score_c161_lexical_echo import infer
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-speex-dialogues"
PLAN = RESULTS / f"{STEM}-plan.json"
SOURCE = RESULTS / "4090-c161-pocketsphinx-willow-v1.json"
ROOT = OUT / "paired-dialogues"
AT = 5 * RATE


def scenes():
    bot_report = read(RESULTS / f"{BOT}-v1.json")
    bank = RESULTS / f"{BOT}-pcm-v1.npz"
    assert sha(bank) == bot_report["pcm_archive_sha256"]
    bot_rows = {r["key"]: r for r in bot_report["records"]}
    with np.load(bank) as clips:
        for index, src in enumerate(read(SOURCE)["records"]):
            speech = load_wav(src)
            bot = ("af_kore_digital", "bm_fable_digital")[index % 2]
            h, room = rir(("h001_", "h003_")[index % 2])
            # At external onset, the panel is 200 ms into "The next train leaves from platform six."
            offset = bot_rows[bot]["spans"][4]["start"] + RATE // 5 - AT
            reference = clips[bot][offset:offset+LENGTH].astype(float)
            echo = fftconvolve(reference, h)[:LENGTH]
            selected = []
            for role, count in (("nyx", 2), ("willow", 1), ("nick", 1), ("max", 1), ("daily", 1)):
                selected.extend([s for s in src["timeline"] if s["role"] == role][:count])
            assert len(selected) == 6 and selected[0]["text"].split(",", 1)[1] == selected[2]["text"].split(",", 1)[1]
            examples = []
            for span in selected:
                clip = speech[span["start"]:span["end"]]
                assert digest(clip) == span["pcm_sha256"] and AT+len(clip) < LENGTH
                ext = np.zeros(LENGTH)
                ext[AT:AT+len(clip)] = clip
                examples.append(ext)
            end = AT + selected[0]["end"] - selected[0]["start"]
            echo *= np.linalg.norm(examples[0][AT:end]) / np.linalg.norm(echo[AT:end])
            scale = min(1., 28000 / max(np.abs(reference).max(), *(abs(s+echo).max() for s in examples),
                                        *(abs(s).max() for s in examples)))
            for j, (span, external) in enumerate(zip(selected, examples, strict=True)):
                end = AT + span["end"] - span["start"]
                sir = 20*np.log10(np.linalg.norm(external[AT:end]) / np.linalg.norm(echo[AT:end]))
                for playing in (False, True):
                    mic = external+echo if playing else external
                    ref = reference if playing else np.zeros_like(reference)
                    mic, ref = (np.rint(x*scale).astype("<i2") for x in (mic, ref))
                    key = f"{src['voice']}-{j}-{int(playing)}"
                    meta = dict(key=key, voice=src["voice"], label=span["role"], text=span["text"],
                                start=AT, end=end, bot=bot, room=room, bot_source_offset=offset,
                                clip_sha256=span["pcm_sha256"], playback=playing, scale=scale,
                                external_to_echo_db=float(sir) if playing else None,
                                mic_sha256=digest(mic), reference_sha256=digest(ref))
                    yield meta, mic, ref


def plan():
    rows = [r for r, _, _ in scenes()]
    assert len(rows) == 48
    cfg = dict(source_sha256=sha(SOURCE), prior_mix_sha256=sha(RESULTS / f"{POST}-mix.json"),
               compare=["single original C161 on linear PCM", "lexical on linear", "lexical on post",
                        "lexical on half", "post and half accepted events in chronological order with half on ties"],
               willow="official DEFAULT on the same half PCM for the first paired request, both playback conditions",
               scope="existing development speech; new acoustic mixes; two source-room assignments", records=rows)
    if PLAN.exists():
        assert read(PLAN) == cfg
    else:
        save(PLAN, cfg)


def outputs(key, mic, ref, archive, builds):
    arrays, summary = {}, {}
    for mode, build in builds.items():
        log = ROOT / f"{key}-{mode}.log"
        receipt = log.with_suffix(".json")
        inputs = dict(mic=digest(mic), ref=digest(ref), elf=sha(build/"nyx_speex_echo.elf"), qemu=sha(QEMU))
        if receipt.exists():
            old = read(receipt)
            assert old["inputs"] == inputs and sha(log) == old["uart"]
        else:
            payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
            boot(build, log, ROOT/"flash.bin", b"SPEEX PASS", (2 << 20, payload))
            parse(log, len(mic))
            save(receipt, dict(inputs=inputs, uart=sha(log)))
        pcm, costs, heap = parse(log, len(mic))
        arrays[mode] = pcm
        summary[mode] = dict(pcm_sha256=digest(pcm), uart_sha256=sha(log), heap=heap,
                             max_us=max(costs), over_10ms=sum(c > 10000 for c in costs))
        archive.write(log, log.name)
    linear = np.r_[np.zeros(LAG, dtype=np.int16), arrays["linear"][:-LAG]]
    arrays["half"] = ((linear.astype(np.int32) + arrays["post"].astype(np.int32)) // 2).astype("<i2")
    return arrays, summary


def score():
    plan()
    root_sources = [read(BASE), read(RESULTS / f"{POST}.json")]
    builds = {"linear": OUT/"build-float", "post": OUT/"build-post"}
    for b, src in zip(builds.values(), root_sources, strict=True):
        assert sha(b/"nyx_speex_echo.elf") == src["elf_sha256"] and sha(QEMU) == src["qemu_sha256"]
    reader, identity = load_reader()
    assert identity == root_sources[1]["asr"]
    mid = RESULTS / read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]["path"]
    assert sha(mid) == root_sources[1]["midlevel_sha256"] and sha(MODEL) == MODEL_SHA == root_sources[1]["c161_sha256"]
    files = LEXICAL/"q8/files"
    manifest = RESULTS/"4090-c161-lexical-p4-q8-packing.json"
    assert sha(manifest) == root_sources[1]["q8_manifest_sha256"]
    assert all(sha(files / n) == v for n, v in read(manifest)["files"].items())
    decoder = Decoder(hmm=str(files/"am"), dict=str(files/"two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    ROOT.mkdir(exist_ok=True)
    wavs = ROOT / "willow-inputs"
    wavs.mkdir(exist_ok=True)
    report = RESULTS / f"{STEM}.json"
    archive = RESULTS / f"{STEM}-uart.zip"
    if report.exists() or archive.exists():
        raise FileExistsError(STEM)
    rows, paired = [], []
    with ZipFile(ROOT/"uart.zip", "w", ZIP_DEFLATED) as uart:
        for meta, mic, ref in scenes():
            key = meta["key"]
            pcm, measurements = outputs(key, mic, ref, uart, builds)
            detections = {name: infer(audio, mid, decoder, cmn) for name, audio in pcm.items()}
            events = {mode: [(e["sample"], mode) for e in v["events"]["both_proposals"]]
                      for mode, v in detections.items()}
            events["c161"] = [(t, "linear") for t in detections["linear"]["events"]["c161"]]
            union = choose([t for t, _ in events["half"]], [t for t, _ in events["post"]])
            events["union"] = [(e["sample"], "half" if e["origin"] == "c161" else "post") for e in union]
            routes, cache = {}, {}
            for route, candidates in events.items():
                hits = []
                for t, mode in candidates:
                    lag = 0 if mode == "linear" else LAG
                    if meta["start"]+lag <= t < meta["end"]+lag+160:
                        hits.append((t, mode))
                record = dict(events=candidates, hits=hits)
                if hits and meta["label"] == "nyx":
                    t, mode = hits[0]
                    if (t, mode) not in cache:
                        end = meta["end"]+(0 if mode == "linear" else LAG)+160
                        first = max(0, (t+159)//160*160-32000)
                        expected = meta["text"].split(",", 1)[1]
                        cache[t, mode] = speech_reading(reader, pcm[mode], first, end, expected)
                    record["capture"] = cache[t, mode]
                routes[route] = record
            if "-0-" in key or meta["label"] == "willow":
                path = wavs / f"{key}.wav"
                sf.write(path, pcm["half"], RATE, subtype="PCM_16")
                paired.append(dict(path=str(path), sha256=sha(path), pcm_sha256=digest(pcm["half"]),
                                   samples=len(mic), target_id=meta["label"], carrier=meta["text"], key=key,
                                   start=meta["start"]+LAG, end=meta["end"]+LAG+160))
            rows.append(dict(key=key, dsp=measurements, half_sha256=digest(pcm["half"]),
                             detections=detections, routes=routes))
            print(key, [(k, bool(v["hits"])) for k, v in routes.items()], flush=True)
    (ROOT/"uart.zip").replace(archive)
    save(report, dict(plan_sha256=sha(PLAN), uart_archive_sha256=sha(archive), asr=identity,
                      linear_sha256=sha(BASE), post_sha256=sha(RESULTS/f"{POST}.json"),
                      records=rows, willow_inputs=paired))


if __name__ == "__main__":
    {"plan": plan, "score": score}[sys.argv[1]]()
