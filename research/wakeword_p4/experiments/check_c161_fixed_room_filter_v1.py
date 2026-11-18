"""Check the frozen fixed-file filter on complete streams and matched cue controls."""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from check_c161_event_transcripts_v1 import event_inputs
from check_c161_measured_room_cues_v1 import room_inputs
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_fixed_room_filter_v1 import STEM, scores
import validate_c161_frozen_cue_v3 as prospective


def digest(pcm):
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def inputs(group):
    if group == "matched":
        for r in read(RESULTS/"4090-c161-v3-command-willow-plan-v1.json")["records"]:
            yield r,load_wav(r)
    elif group == "rooms":
        prior = read(RESULTS/"4090-c161-measured-room-cues-v1.json")["records"]
        for (r,pcm,_),old in zip(room_inputs(),prior,strict=True):
            assert r["pcm_sha256"] == old["pcm_sha256"]
            if r["cue"] == "v3":
                yield dict(**r,prior=old["confirmed"]),pcm
    elif group == "music":
        for r in read(RESULTS/"4090-c161-music-development-sources-v1.json")["records"]:
            yield r,load_wav(r)
    else:
        for r,_,raw,pcm in event_inputs(group):
            meta = {k:v for k,v in r.items() if k not in ("events","physical_calls","expanded_calls")}
            yield dict(**meta,baseline=raw["c161"]),pcm


def analyze(weights,pcm):
    codes,clock = scores(pcm,weights)
    return dict(events=times(decode(codes,clock)), peak5=int(np.convolve(codes.astype(np.int32),np.ones(5,np.int32),"valid").max()),
                scores_sha256=hashlib.sha256(codes.tobytes()).hexdigest())


def old_prospective(weights):
    prior = read(prospective.OUT)["records"]
    rows = []
    def score_current(lib,chunks):
        pcm, = chunks
        old = prior[len(rows)]
        assert digest(pcm) == old["pcm_sha256"]
        rows.append(dict(source=old,**analyze(weights,pcm)))
        return old["confirmed"]
    with tempfile.TemporaryDirectory() as temp, patch.object(prospective,"OUT",Path(temp)/"identity.json"), \
         patch.object(prospective.matcher,"LIB",Path(temp)/"unused.so"), patch.object(prospective.matcher,"binding"), \
         patch.object(prospective.matcher,"detect",score_current):
        prospective.run()
        assert read(prospective.OUT)["records"] == prior
    return rows


def main(groups):
    training = read(RESULTS/f"{STEM}-training-v1.json")
    path = RESULTS/f"{STEM}-weights-v1.npz"
    assert sha(path) == training["weights_sha256"]
    with np.load(path) as weights:
        for group in groups:
            if group == "prospective":
                rows = old_prospective(weights)
            else:
                rows = []
                for r,pcm in inputs(group):
                    assert digest(pcm) == r["pcm_sha256"]
                    rows.append(dict(source=r,**analyze(weights,pcm)))
            save(RESULTS/f"{STEM}-{group}-v1.json",dict(weights_sha256=sha(path),records=rows))
            print(group,len(rows),sum(len(r["events"]) for r in rows),flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
