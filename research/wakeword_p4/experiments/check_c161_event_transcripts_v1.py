"""Check candidate-only notifications against an independent GPU ASR on bounded past audio."""
import hashlib
import os
import re
from pathlib import Path

os.environ["USE_TF"] = "0"
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from check_c161_bounded_transfer_v1 import sources as transfer_sources
from check_c161_bot_replies_v1 import STEM as BOT
from check_c161_pronunciation_ctc_v1 import MODEL
from check_c161_stream_stretch_v1 import STEM as STRETCH
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import cases
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-event-transcripts"
WINDOW = 19200
DELAYS = (0, 1920)
PHRASE = re.compile(r"\b(?:HEY|HAY)\s*(?:NICKS|NIX|NYX|NIKS)\b")


def source_pcm(cohort):
    if cohort == "bot":
        meta = read(RESULTS / f"{BOT}-v1.json")
        path = RESULTS / f"{BOT}-pcm-v1.npz"
        assert sha(path) == meta["pcm_archive_sha256"]
        with np.load(path) as data:
            for row in meta["records"]:
                yield row["key"], data[row["key"]]
    elif cohort == "transfer":
        for row, pcm in transfer_sources():
            yield row["group"] + "_" + row["key"], pcm
    elif cohort == "forward":
        for i, row in enumerate(read(RESULTS / "4090-c161-signed-forward-v1.json")["records"]):
            yield str(i), load_wav(row)
    else:
        for row, pcm in cases(cohort):
            yield row["key"], pcm


def choose(baseline, confirmed):
    # Each decoder already implements its per-model re-arm and cooldown.
    # Merge the accepted notifications in physical clock order, preferring C161 on a tie.
    eligible = [(t, "c161") for t in baseline]
    eligible.extend((t, "verified") for t in confirmed)
    emitted = []
    for t, origin in sorted(eligible):
        if not emitted or (t + 479)//480 - (emitted[-1]["sample"] + 479)//480 >= 67:
            emitted.append(dict(sample=t, origin=origin))
    return emitted


def event_inputs(cohort):
    path = RESULTS / (f"{BOT}-v1.json" if cohort == "bot" else f"{STRETCH}-{cohort}-v1.json")
    rows = read(path)["records"]
    for row, (key, pcm) in zip(rows, source_pcm(cohort), strict=True):
        record_key = row["group"]+"_"+row["key"] if cohort == "transfer" else row["key"]
        assert key == record_key and hashlib.sha256(pcm.tobytes()).hexdigest() == row["pcm_sha256"]
        if cohort == "context" and "timeline" not in row:
            continue
        if cohort == "bot":
            events = row["events"]
        else:
            ref = row["events"]
            events = dict(c161=ref["c161"]["identity"], midlevel=ref["midlevel"]["identity"],
                          expanded=ref["midlevel"]["stretch"])
        yield row, key, events, pcm


def main():
    torch.set_num_threads(2)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    identity = read(MODEL / "source.json")
    lock = read(RESULTS / "4090-ctc-training-audit-plan-v1.json")
    assert identity["files"]["model.safetensors"]["sha256"] == lock["expected_safetensors_sha256"]
    assert all(sha(MODEL / name) == f["sha256"] for name, f in identity["files"].items())
    processor = Wav2Vec2Processor.from_pretrained(MODEL, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    reports = []
    with torch.inference_mode():
        for cohort in ("context", "forward", "transfer", "daily", "joe", "mac", "bot"):
            for row, key, raw, pcm in event_inputs(cohort):
                proposed = sorted(set(raw["midlevel"] + raw["expanded"]))
                ends = sorted({t+d for t in proposed for d in DELAYS if 0 < t+d <= len(pcm)})
                windows, readings = [], {}
                for end in ends:
                    crop = pcm[max(0, end-WINDOW):end]
                    windows.append(np.pad(crop.astype(np.float32)/32768, (WINDOW-len(crop), 0)))
                if ends:
                    batch = processor(windows, sampling_rate=16000, return_tensors="pt").input_values.cuda()
                    ids = model(input_values=batch).logits.argmax(-1).cpu()
                    for end, transcript in zip(ends, processor.batch_decode(ids), strict=True):
                        readings[end] = dict(end=end, transcript=transcript, accepted=bool(PHRASE.search(transcript)))
                merged = {}
                for delay in DELAYS:
                    confirmed = [t+delay for t in proposed if readings.get(t+delay, {}).get("accepted")]
                    merged[str(delay)] = choose(raw["c161"], confirmed)
                meta = {k:v for k,v in row.items() if k not in ("events", "physical_calls", "expanded_calls")}
                reports.append(dict(cohort=cohort, record_key=key, meta=meta, samples=len(pcm),
                                    raw=raw, readings=list(readings.values()), merged=merged))
            print(cohort, flush=True)
    save(RESULTS / f"{STEM}-v1.json", dict(gpu=gpu, torch=torch.__version__, asr=identity, window=WINDOW,
         delays=DELAYS, phrase_pattern=PHRASE.pattern, input_reports={c:sha(RESULTS / f"{STRETCH}-{c}-v1.json")
         for c in ("context","forward","transfer","daily","joe","mac")}, bot_report=sha(RESULTS/f"{BOT}-v1.json"),
         records=reports))


if __name__ == "__main__":
    main()
