"""Interleave nominal- and multirate-pattern groups across the fixed source duration."""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

import bella_nominal_first as experiment

P, fit = experiment.P, experiment.fit
ROOT = experiment.ROOT
WORK = Path(".local/wakeword/bella-third-triplet")
OUT = experiment.OUT.with_name("4090-bella-third-triplet-fit.json")
PROTO = experiment.PROTO.with_name("bella_third_triplet_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
ORDERS = ("nominal-before-A", "multirate-before-nominal")


def wave(name):
    return OUT.with_name(f"hye-nyx-bella-{name}.wav")


def build():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    seed = P.read(fit.early.playback.OUT)
    base_path = fit.early.WAV.with_name(seed["cue"])
    base,sr = sf.read(base_path,dtype="int16")
    a,ar = sf.read(fit.early.WAV,dtype="int16")
    nominal_path=experiment.wav_path("nominal")
    nominal,nr=sf.read(nominal_path,dtype="int16")
    assert sr==ar==nr==16000 and P.sha(base_path)==seed["cue_sha256"]
    assert P.sha(nominal_path)==P.read(experiment.OUT)["voices"]["nominal"] and P.sha(fit.early.WAV)==P.read(fit.early.FIT)["cue_sha256"]
    docs, hashes={},{}
    for name,carrier,donor in zip(ORDERS,(a,nominal),(nominal,a),strict=True):
        delta=donor.astype(float)-base
        assert not np.any(delta[:6464]) and not np.any(delta[-64:])
        moved=np.roll(delta,-6400)
        old=torch.tensor(carrier,device="cuda",dtype=torch.float64)
        extra=torch.tensor(moved,device="cuda",dtype=torch.float64)
        trials=[]
        for shift in range(-64,65,4):
            for sign in (1,-1):
                v=sign*torch.roll(extra,shift)
                pos,neg=v>0,v<0
                gain=min(1.,float(((32767-old[pos])/v[pos]).min()),float(((-32768-old[neg])/v[neg]).min()))
                trials.append(dict(shift=shift,sign=sign,gain=gain))
        best=min(trials,key=lambda r:(-r["gain"],abs(r["shift"]),-r["sign"],r["shift"]))
        v=best["sign"]*np.roll(moved,best["shift"])
        assert not np.any((v!=0)&(carrier!=base))
        total=np.rint(carrier.astype(float)+best["gain"]*v)
        assert total.max()<=32767 and total.min()>=-32768
        pcm=total.astype("<i2")
        assert np.array_equal(pcm[v==0],carrier[v==0])
        blob=fit.early.playback.playback.sources.wav(pcm,"PCM_16")
        wave(name).write_bytes(blob)
        hashes[name]=P.sha(wave(name))
        docs[name]=dict(winner=best,trials=trials,samples=len(pcm),changed=int(np.count_nonzero(pcm!=carrier)),
                        first_change=int(np.flatnonzero(pcm!=carrier)[0]),last_change=int(np.flatnonzero(pcm!=carrier)[-1]))
    deps=(SCRIPT,PROTO,experiment.SCRIPT,experiment.OUT,nominal_path,fit.early.WAV,base_path)
    P.save(OUT,dict(voices=hashes,orders=docs,gpu=torch.cuda.get_device_name(),sources={str(p):P.sha(p) for p in deps}))


def replay(name):
    # The shared 59-case runner uses its experiment paths for evidence and caches.
    from unittest.mock import patch
    with patch.multiple(experiment, OUT=OUT, WORK=WORK, SCRIPT=SCRIPT, PROTO=PROTO, wav_path=wave):
        experiment.replay(name)


def history(name):
    import check_bella_headroom_history as historical
    from unittest.mock import patch
    folder=WORK/name
    receipt=folder/"history-cue.json"
    P.save(receipt,dict(fit=dict(cue_sha256=P.read(OUT)["voices"][name])))
    history_out=OUT.with_name(f"4090-bella-third-triplet-{name}-history.json")
    with patch.multiple(fit,WAV=wave(name),WORK=folder,OUT=receipt), patch.object(historical,"OUT",history_out):
        historical.main()


if __name__ == "__main__":
    if sys.argv[1]=="build":
        build()
    elif sys.argv[1] in ORDERS:
        replay(sys.argv[1])
    elif sys.argv[1]=="history" and sys.argv[2] in ORDERS:
        history(sys.argv[2])
