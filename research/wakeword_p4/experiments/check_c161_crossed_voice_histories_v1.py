"""Cross intact target utterances with each preceding voice and an equal-length silent history."""
import hashlib
from unittest.mock import patch

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-crossed-voice-histories"


def main():
    fit = read(RESULTS / "4090-c161-protected-confuser-training-v1.json")
    model = RESULTS / fit["reports"][0]["path"]
    assert sha(model) == fit["reports"][0]["sha256"]
    native = binding()
    rows = []
    for group, name, prior_name in (
            ("first", "4090-c161-four-voice-forward", "4090-c161-protected-confuser-transfer-v1.json"),
            ("second", "4090-c161-signed-forward", "4090-c161-protected-confuser-forward-v1.json")):
        data = read(RESULTS / f"{name}-plan-v1.json")
        items = [(r,load_wav(r)) for r in data["records"] if r["target_id"] == "nyx"]
        baseline = {(r["voice"],r["scene"]):r for r in read(RESULTS/prior_name)["records"]
                    if r.get("role",r.get("target")) == "nyx"}
        with patch.object(live, "MODEL", model):
            for target, pcm in items:
                voice, scene = target["voice"], target["scene"]
                utterance = pcm[target["start"]:]
                wake = pcm[target["start"]:target["first_phone"]]
                previous = [(h["voice"], p[:h["start"]]) for h,p in items if h["scene"] == scene]
                previous.append(("silence",np.zeros(target["start"],np.int16)))
                for context, prefix in previous:
                    full = np.r_[prefix, utterance]
                    onset = len(prefix)
                    first = onset + len(wake)
                    end = onset + target["end"] - target["start"]
                    codes, clock, pair, _ = live.run_stream(native, full)
                    clock = np.asarray(clock)
                    primary = np.asarray(codes[0],np.uint8)
                    events = times(decode(primary, clock))
                    if context == voice:
                        assert hashlib.sha256(full.tobytes()).hexdigest() == target["pcm_sha256"]
                        assert baseline[(voice,scene)]["primary"] == events
                        assert baseline[(voice,scene)]["pair"] == times(pair)
                    sums = np.convolve(primary.astype(int),np.ones(5,int),"valid")
                    # Include up to 240 ms after the first command phone, sufficient for prior positive timing.
                    eligible = np.flatnonzero((clock[:-4] > onset) & (clock[4:] <= min(end,first+3840)))
                    j = eligible[sums[eligible].argmax()]
                    rows.append(dict(group=group,scene=scene,target_voice=voice,history_voice=context,
                                     target_sha256=hashlib.sha256(utterance.tobytes()).hexdigest(),
                                     pcm_sha256=hashlib.sha256(full.tobytes()).hexdigest(),start=onset,end=end,first_phone=first,
                                     wake_rms=float(np.sqrt(np.mean(wake.astype(float)**2))),
                                     primary=events,pair=times(pair),peak5=int(sums[j]),peak_end=int(clock[j+4])))
    assert len(rows) == 120
    save(RESULTS/f"{STEM}-v1.json",dict(model_sha256=sha(model),records=rows))


if __name__ == "__main__":
    main()
