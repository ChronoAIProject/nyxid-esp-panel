"""Compare synthetic positives at the annotated word tail and at the live INT8 proposal."""
import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np

import train_proposal_readout as parent
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, sha

STEM = "4090-c161-positive-readout"
FIT = RESULTS/(STEM+"-fit.json")
MODES = ("aligned","trigger")
POSITIVE_POLICY = ("In both fits use the same peak negatives and 16 additional positive samples per batch. "
                   "Pair the same training-only microphone input at its annotated word tail plus 360 ms and "
                   "at the first live word proposal within +/-240 ms of that tail plus 360 ms. "
                   "Use one clean and one frozen training-room view per eligible source; log unproposed views.")


def weights(mode):
    return RESULTS/f"{STEM}-{mode}.npz"


def collect():
    word = parent.music.word
    inventory = read(RESULTS/(word.STEM+"-training-v1.json"))
    by_key = {r["key"]:r for r in inventory["training"]}
    front = RESULTS/(parent.music.STEM+"-v1.tflite")
    vm = REF_KERNEL(model_path=str(front))
    vm.allocate_tensors()
    lib = parent.replay.BUILD/"frontend.so"
    assert sha(lib) == read(parent.replay.BUILD/"build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    augment = parent.music.mining.room_render()
    rows = []
    views = {m:[] for m in MODES}
    for src, wave in word.sources():
        r = by_key.pop(src["key"])
        assert r["pcm_sha256"] == parent.replay.poll.digest(wave) and r["positive"] == src["positive"]
        if not src["positive"]:
            continue
        onset = 32000
        dry = np.pad(wave,(onset,16000))
        wet, setting = augment(src,dry,onset)
        tail = onset+src["end"]
        for condition, pcm in (("clean",dry),("room",wet)):
            if condition == "room" and np.array_equal(dry,wet):
                continue
            raw, bands, clock = parent.replay.native_inputs(native,pcm)
            codes = parent.replay.infer(vm,bands)
            runtime = parent.replay.scoring.word
            ts = runtime.times(runtime.decode(codes,clock))
            t = next((t for t in ts if onset<=t and abs(t-tail)<=3840 and t+runtime.WAIT<=len(pcm)),None)
            row = dict(key=src["key"], voice=src["voice"], text=src["text"], condition=condition, tail=tail,
                       source_pcm_sha256=r["pcm_sha256"], pcm_sha256=parent.replay.poll.digest(pcm), proposals=ts,
                       accepted=t, room=setting if condition == "room" else None)
            if t is not None:
                for mode, end in (("trigger",t+runtime.WAIT),("aligned",(tail+runtime.WAIT)//160*160)):
                    patch = parent.room.window(raw,end)
                    x = np.rint(patch.clip(0,666)*(255./666)).astype(np.uint8)
                    views[mode].append(x)
                    row[mode] = dict(end=end,window_sha256=parent.replay.poll.digest(x))
            rows.append(row)
    assert not by_key
    return {k:np.stack(v) for k,v in views.items()},rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("train","export","score"))
    p.add_argument("groups",nargs="*")
    args = p.parse_args()
    experiment = sys.modules[__name__]
    if args.action == "train":
        parent.train(experiment,collect)
    elif args.action == "export":
        from export_c161_polling_rooms import main
        main(experiment)
    else:
        import score_proposal_readout as scoring
        for group in args.groups:
            if group not in scoring.GROUPS:
                raise ValueError(group)
            scoring.main(group,experiment)
