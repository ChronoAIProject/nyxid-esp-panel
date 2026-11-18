"""Route the alternate readout after a completed speech interval; evaluate past-only gating."""
from unittest.mock import patch
import hashlib

import numpy as np
import webrtcvad

from audit_c161_history_paths_v1 import arbitrate, decode, times
from fit_c161_history_gain_v1 import MODEL, PRIME, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding, cases
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-paired-event-readout"


def history_gate(pcm, clock):
    # A 300 ms voiced run followed by at least 200 ms of VAD silence qualifies
    # the next three seconds. All markers are hop ends measured on the PCM clock.
    vad = webrtcvad.Vad(0)
    full = np.r_[np.zeros(PRIME, np.int16), pcm]
    hops = np.pad(full, (0, -len(full) % 160)).reshape(-1, 160)
    voiced = quiet = 0
    completed = []
    for i, frame in enumerate(hops):
        if vad.is_speech(frame.tobytes(), 16000):
            if quiet:
                voiced = 0
            voiced += 1
            quiet = 0
        else:
            quiet += 1
            if quiet == 20 and voiced >= 30:
                completed.append((i+1)*160 - PRIME)
    on = np.zeros(len(clock), dtype=bool)
    for t in completed:
        on |= (clock >= t) & (clock < t + 48000)
    return on, completed


def events(pcm, clock, original, speech):
    active, ends = history_gate(pcm, clock)
    first = decode(original[0], clock)
    result = {}
    for name, alt in (("base", original[1]), ("always", speech[1]),
                      ("conditional", np.where(active, speech[1], original[1]))):
        result[name] = times(arbitrate(first, decode(alt, clock))[0])
    return result, ends


def cohorts():
    rows = []
    for cohort in ("context", "willow", "daily", "joe", "mac", "generated"):
        report = read(RESULTS / f"{STEM}-{cohort}-v1.json")
        path = RESULTS / f"{STEM}-{cohort}-scores-v1.npz"
        assert sha(path) == report["scores_sha256"]
        with np.load(path) as scores:
            for (meta, pcm), item in zip(cases(cohort), report["records"], strict=True):
                k = meta["key"]
                assert item["key"] == k and item["pcm_sha256"] == hashlib.sha256(pcm.tobytes()).hexdigest()
                hits, ends = events(pcm, scores[k+"_clock"], scores[k+"_c161"], scores[k+"_speech"])
                assert hits["base"] == item["events"]["c161"]["pair"]
                rows.append(dict(cohort=cohort, key=k, pcm_sha256=item["pcm_sha256"], gates=ends, events=hits))
    save(RESULTS / "4090-c161-speech-history-gate-v1.json", dict(records=rows))


def conversation_daily():
    # Distinct text contexts across the 96 consumed everyday negatives, no source selection by model.
    prefix_pool = [r for r in read(RESULTS / "4090-chatterbox-daily-training-admission-v1.json")["admitted"]
                   if r["request"]["group"] == "conversation"]
    model_info = read(RESULTS / f"{STEM}-training-v1.json")
    m, = [r for r in model_info["reports"] if r["name"] == "speech"]
    candidate = RESULTS / m["path"]
    assert sha(candidate) == m["sha256"] and sha(MODEL) == model_info["baseline_sha256"]
    native = binding()
    rows = []
    for i, (meta, target) in enumerate(cases("daily")):
        prefix = prefix_pool[i % len(prefix_pool)]
        gap = (4000, 8000, 16000)[i % 3]
        lead = load_wav(prefix["wav"])
        pcm = np.r_[lead, np.zeros(gap, np.int16), target]
        base, clock, base_events, _ = live.run_stream(native, pcm)
        with patch.object(live, "MODEL", candidate):
            trained, candidate_clock, _, _ = live.run_stream(native, pcm)
        assert clock == candidate_clock
        hits, ends = events(pcm, np.asarray(clock), np.asarray(base, np.uint8), np.asarray(trained, np.uint8))
        assert hits["base"] == times(base_events)
        rows.append(dict(key=meta["key"], text=meta["text"], gap=gap, prefix_id=prefix["index"],
                         target_start=len(lead)+gap, samples=len(pcm), gates=ends, events=hits,
                         pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    save(RESULTS / "4090-c161-speech-history-daily-conversations-v1.json", dict(records=rows))


if __name__ == "__main__":
    cohorts()
    conversation_daily()
