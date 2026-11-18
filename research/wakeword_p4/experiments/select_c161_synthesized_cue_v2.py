"""Select an intact training-only TTS recording, then test fixed playback perturbations."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from audit_c161_history_paths_v1 import decode, times
from generate_c161_vowel_name_pair_v1 import RESULTS, sha
from replay_c161_vowel_name_pair_v1 import binding
from review_c161_equal_length_history_v1 import MODEL, MODEL_SHA
import verify_c161_ambient_stream_v1 as live

LOCK = RESULTS / "4090-c161-synthetic-cue-selection-v2.json"
WAV = RESULTS / "hye-nyx-c161-synthetic-cue-v2.wav"


def candidates():
    piper = json.loads((RESULTS / "4090-multispeaker-coverage-native-original-v1.json").read_bytes())
    ledger = Path(piper["records_path"])
    assert sha(ledger) == piper["records_sha256"]
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row["text"] == "Hey Nicks." and row["length_scale"] == 1 and row["draw"] == 0:
            assert row["role"] == "training"
            yield dict(engine=row["voice"], voice=row["speaker_name"], path=str(ledger.parent / row["path"]), sha256=row["sha256"])
    ledger = Path(".local/wakeword/hey-broad-kokoro-native-v1/records.jsonl")
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        r = row["request"]
        if r["text"] == "Hey Nicks." and r["speed"] == 1:
            assert r["split"] == "training" and row["quality_pass"]
            wav = dict(row["files"]["wav"])
            wav["path"] = str(ledger.parent / wav["path"])
            yield dict(engine="kokoro", voice=r["conditioning_id"], **wav)


def load_wav(row):
    path = Path(row["path"])
    assert sha(path) == row["sha256"]
    samples, rate = sf.read(path, dtype="int16")
    assert rate == 16000 and samples.ndim == 1
    return samples


def trial(native, cue, prefix, gap, gain_db, snr, seed, room=False, color=False):
    source = cue.astype(float)
    if room:
        # One direct path plus two delayed reflections; carry the full echo tail.
        h = np.zeros(721)
        h[[0, 240, 720]] = [1, .22, -.12]
        source = np.convolve(source, h)
    start = len(prefix) + gap
    full = np.r_[prefix, np.zeros(gap), source * 10**(gain_db/20), np.zeros(8000)]
    if snr is not None:
        noise = np.random.default_rng(seed).normal(size=len(full))
        if color:
            noise = lfilter([1], [1, -.9], noise)
        rms = np.sqrt(np.mean(cue.astype(float)**2))
        full += noise / np.sqrt(np.mean(noise**2)) * rms / 10**(snr/20)
    pcm = np.rint(full).clip(-32768, 32767).astype("<i2")
    codes, clock, emitted, _ = live.run_stream(native, pcm)
    scores = np.asarray(codes, np.uint8)
    clock = np.asarray(clock)
    ev = times(decode(scores[0], clock))
    pair = times(emitted)
    within = (clock[4:] > start) & (clock[4:] <= start + len(cue))
    sums = np.convolve(scores[0].astype(int), np.ones(5, dtype=int), "valid")
    return dict(start=start, end=start+len(cue), original=ev, pair=pair,
                score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), clipped=int(np.count_nonzero((full>32767)|(full< -32768))),
                max5=int(sums[within].max()), original_in_cue=any(start<t<=start+len(cue) for t in ev),
                pair_in_cue=any(start<t<=start+len(cue) for t in pair))


def select():
    from faster_whisper import WhisperModel
    model_path = Path(".local/wakeword/whisper-small-en")
    plan = json.loads((RESULTS/"4090-parallel-melo-context-asr-plan-v1.json").read_bytes())
    for file in model_path.iterdir():
        if file.is_file():
            assert sha(file) == plan["files"][str(file)]
    asr = WhisperModel(str(model_path), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1)
    native = binding()
    rows = list(candidates())
    assert len(rows) == 50 and len({r["sha256"] for r in rows}) == 50
    for r in rows:
        cue = load_wav(r)
        segments, _ = asr.transcribe(cue.astype(np.float32)/32768, language="en", beam_size=3,
                                     temperature=0, vad_filter=False, condition_on_previous_text=False)
        r["transcript"] = "".join(s.text for s in segments).strip()
        r["equivalent"] = re.findall("[a-z]+",r["transcript"].lower()) in [
            ["hey", word] for word in ("nyx","nix","nicks","knicks")]
        r["trials"] = [trial(native,cue,np.zeros(16000,np.int16),4000,0,snr,3501) for snr in (None,20,12)]
        r["hits"] = sum(v["original_in_cue"] for v in r["trials"])
        r["worst_max5"] = min(v["max5"] for v in r["trials"])
    eligible = [r for r in rows if r["equivalent"] and r["hits"] and all(
        all(t>v["start"] for t in v["original"]) for v in r["trials"])]
    chosen = sorted(eligible, key=lambda r:(-r["hits"],-r["worst_max5"],r["sha256"]))[0] if eligible else None
    with LOCK.open("x") as f:
        json.dump(dict(model_sha256=MODEL_SHA, chosen=chosen, candidates=rows), f, indent=2)
        f.write("\n")
    if chosen:
        shutil.copyfile(chosen["path"], WAV)
        assert sha(WAV) == chosen["sha256"]


def test():
    lock = json.loads(LOCK.read_bytes())
    chosen = lock["chosen"]
    if not chosen:
        return
    assert sha(WAV) == chosen["sha256"]
    native = binding()
    admission = json.loads((RESULTS/"4090-chatterbox-daily-training-admission-v1.json").read_bytes())
    voices = ("af_bella","am_onyx","bf_isabella","bm_fable")
    records = []
    for label, path in [("v1", RESULTS/"hye-nyx-c161-synthetic-cue-v1.wav"), ("v2", WAV)]:
        cue, rate = sf.read(path, dtype="int16")
        assert rate == 16000
        for i, voice in enumerate(voices):
            choices = [r for r in admission["admitted"] if r["request"]["voice"] == voice and r["request"]["group"] == "ordinary_request"]
            row = choices[i]
            prefix = load_wav(row["wav"])
            for gap in (3040, 6880):
                for room in (False, True):
                    for gain in (0,-6):
                        for snr in (None,20,12):
                            seed = 57638+i*1000+gap
                            res = trial(native,cue,prefix,gap,gain,snr,seed,room,color=True)
                            records.append(dict(cue=label, history=row["request"]["text"], history_sha256=row["wav"]["sha256"],
                                                voice=voice, gap=gap, room=room, gain_db=gain, nominal_snr=snr, **res))
    with (RESULTS/"4090-c161-synthetic-cue-validation-v2.json").open("x") as f:
        json.dump(dict(selection_sha256=sha(LOCK), wav_sha256=sha(WAV), records=records),f,indent=2)
        f.write("\n")


if __name__ == "__main__":
    assert sha(MODEL) == MODEL_SHA
    {"select":select, "test":test}[sys.argv[1]]()
