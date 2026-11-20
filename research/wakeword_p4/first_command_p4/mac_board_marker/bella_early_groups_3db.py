"""Freeze the two-group gain change before measuring the receiver."""
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import soundfile as sf
import torch

import bella_third_triplet as triple

P = triple.P
NAME = "early-groups-plus3db"
WORK = Path(".local/wakeword/bella-early-groups-3db")
PROTO = triple.PROTO.with_name("bella_early_groups_3db_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
WAV = triple.wave(NAME)
FIT = triple.OUT.with_name("4090-bella-early-groups-3db-fit.json")
OLD = triple.wave("multirate-before-nominal")
SINGLE = OLD.with_name("hye-nyx-af-bella-plus5db.wav")
REPLAY = FIT.with_name(f"4090-bella-nominal-first-{NAME}.json")


def build():
    assert P.sha(OLD) == P.read(triple.OUT)["voices"]["multirate-before-nominal"]
    assert P.sha(SINGLE) == P.read(triple.OUT)["sources"][str(SINGLE)]
    base, sr = sf.read(SINGLE, dtype="int16")
    previous, rr = sf.read(OLD, dtype="int16")
    assert sr == rr == 16000 and previous.shape == base.shape == (23200,)
    name = torch.cuda.get_device_name()
    assert "4090" in name
    single = torch.tensor(base, dtype=torch.float64, device="cuda")
    delta = torch.tensor(previous, dtype=torch.float64, device="cuda") - single
    gain = 10**(3/20)
    total = (single + gain*delta).round().cpu().numpy()
    assert total.min() >= -32768 and total.max() <= 32767
    pcm = total.astype("<i2")
    assert np.array_equal(previous[15518:], pcm[15518:])
    WAV.write_bytes(triple.fit.early.playback.playback.sources.wav(pcm,"PCM_16"))
    deps = (OLD, SINGLE, triple.OUT, SCRIPT, PROTO)
    P.save(FIT,dict(voices={NAME:P.sha(WAV)}, pcm_sha256=P.digest(pcm), gain=gain, samples=len(pcm),
                    changed=int(np.count_nonzero(pcm != previous)), minimum=int(pcm.min()), maximum=int(pcm.max()),
                    gpu=name, torch=str(torch.__version__), sources={str(p):P.sha(p) for p in deps}))


def replay():
    with patch.multiple(triple.experiment, OUT=FIT, WORK=WORK, SCRIPT=SCRIPT, PROTO=PROTO,
                        wav_path=lambda name: WAV):
        triple.experiment.replay(NAME)


if __name__ == "__main__":
    {"build":build, "replay":replay}[sys.argv[1]]()
