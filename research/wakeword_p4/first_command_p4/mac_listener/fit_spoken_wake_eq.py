"""Fit a single causal microphone EQ for the frozen C161 detector across synthetic voices."""
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, firwin2, lfilter

import select_room_spoken_cues as selection

ch, room = selection.ch, selection.room
STEM = "4090-c161-spoken-wake-eq"
FIT = ch.RESULTS / (STEM+"-fit.json")
FREQ = (0,125,250,500,1000,2000,4000,6000,8000)
SHAPES = {
    "low_cut": (-18,-18,-9,-3,0,0,0,0,0),
    "presence": (-18,-14,-8,-2,0,4,4,0,-6),
    "treble": (-4,-4,-3,-1,0,3,6,6,6),
    "mid_band": (-20,-16,-8,-2,0,0,-3,-12,-18),
    "high_cut": (0,0,0,0,0,-3,-9,-12,-12),
}


def taps():
    bank = {"raw":np.array([16384],dtype=int)}
    for name,db in SHAPES.items():
        b = firwin2(65, FREQ, 10**(np.asarray(db)/20), fs=16000)
        response = abs(b @ np.exp(-2j*np.pi*1000/16000*np.arange(len(b))))
        bank[name] = np.rint(16384*b/response).astype(int)
    return bank


def process(wave, coeff):
    if len(coeff) == 1:
        assert coeff[0]==16384
        return wave,0
    # All products and sums of these small integers are exact in float64.
    acc = lfilter(coeff.astype(float),[1.],wave)
    q = np.floor((acc+8192.)/16384.)
    clips = int(np.count_nonzero((q>32767)|(q< -32768)))
    return q.clip(-32768,32767).astype("<i2"), clips


def main():
    frozen = ch.read(selection.FIT)
    selected = frozen["choices"]
    assert frozen["pool_sha256"]==ch.sha(selection.PACK)
    native = room.rooms.c161_binding()
    recipes = list(selection.mixtures())
    bank = {r["index"]:x for r,x in selection.pool()}
    filters = taps()
    rows, totals, per_voice = [], {n:Counter() for n in filters}, {}
    for voice,choices in selected.items():
        per_voice[voice] = {n:Counter() for n in filters}
        for j,(condition,h,bg) in enumerate(recipes):
            end = condition["end"]
            arms = {n:{} for n in filters}
            for role in ("target","nick"):
                i = choices["selected"][role]
                cue = bank[i]
                start = end-len(cue)
                wet = fftconvolve(cue.astype(float),h)
                mix = bg.copy()
                count = min(len(wet),len(mix)-start)
                mix[start:start+count] += wet[:count]
                mic = room.quantize(mix)
                before = frozen["records"][i]["tests"][j]
                assert ch.digest(mic)==before["pcm_sha256"]
                for name,b in filters.items():
                    wave,clipped = process(mic,b)
                    test = room.measure(native,wave,start,end)
                    clocks = [selection.gap.study.voices.timing.block(t) for t in test["events"]]
                    timely = bool(clocks and start<=clocks[0]<end)
                    if name=="raw":
                        assert test["events"]==before["events"]
                    arms[name][role] = dict(first_open=next(iter(clocks),None),timely=timely,
                                             peak5=test["peak5"],clipped=clipped,pcm_sha256=ch.digest(wave))
            for name,out in arms.items():
                good = out["target"]["timely"] and out["nick"]["first_open"] is None
                counts = dict(paired=good,timely=out["target"]["timely"],nick=out["nick"]["first_open"] is not None)
                totals[name].update(counts)
                per_voice[voice][name].update(counts)
            rows.append(dict(voice=voice,condition=j,routes=arms))
    def rank(name):
        t = totals[name]
        return (-t["paired"],t["nick"],-min(v[name]["paired"] for v in per_voice.values()),name)
    best = min(SHAPES,key=rank)
    ch.save(FIT,dict(records=rows,selected=best,totals=totals,per_voice=per_voice,conditions=[r for r,_,_ in recipes],
                     taps={n:b.tolist() for n,b in filters.items()},shapes=SHAPES,frequency_hz=FREQ,
                     sources={p.name:ch.sha(p) for p in (selection.FIT,selection.PACK,room.rooms.MODEL,Path(__file__))},
                     policy="Same eight previously selected synthetic voices, both names, three training rooms "
                     "and dry control. Select one shared 65-tap causal Q14 filter by timely Nyx and no Nick, "
                     "then lower Nick count, then worst-voice paired score. Compare against raw even when raw "
                     "wins fit. Feed the full stream to frozen C161, retaining 32 samples of group delay. "
                     "Uplink stays on the original microphone."))


if __name__ == "__main__":
    main()
