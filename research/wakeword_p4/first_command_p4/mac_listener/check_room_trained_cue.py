"""Paired playback level and clock sweep on separate room responses using the untouched listener."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_room_noise_frontends as parent
import handoff_clock_rates as drift
import train_room_cue as fit

ch, room, gap = parent.ch, parent.room, parent.gap
OUT = ch.RESULTS / (fit.STEM + "-grid.json")
LEVELS = (0,-6,-12,-18)
PPM = (-1000,0,1000)


def scenes():
    report = ch.read(fit.FIT)
    original, sr = sf.read(fit.fixed.CUE,dtype="int16")
    candidate, new_sr = sf.read(fit.WAV,dtype="int16")
    assert sr==new_sr==16000 and ch.sha(fit.WAV)==report["wav_sha256"]
    markers = {name:{ppm:drift.project(wav, Fraction(1000000,1000000+ppm)) for ppm in PPM}
               for name,wav in (("v5",original),("trained",candidate))}
    bank = room.rooms.source_bank()
    commands = sorted({(s["voice"],s["carrier"]) for s in ch.read(room.rooms.PLAN)["records"]})
    noises = {kind:room.rooms.read_noise(kind) for kind in ("music","air_conditioner")}
    seen = {r["sha256"] for r in report["training_rooms"]}
    end = 51200
    for ri, number in enumerate(report["development_room_ids"]):
        h, location, rh = room.rooms.read_room(f"h{number:03}_")
        assert rh not in seen
        wet = {name:{ppm:fftconvolve(a.astype(float),h) for ppm,a in waves.items()}
               for name,waves in markers.items()}
        baseline = wet["v5"][0][:len(original)]
        reference_rms = np.sqrt(np.mean(baseline**2))
        peak = max(float(np.abs(a).max()) for waves in wet.values() for a in waves.values())
        for bi, (kind,(noise,path,nh)) in enumerate(noises.items()):
            voice,command = commands[(ri+3*bi)%len(commands)]
            cmd = bank[voice,command]
            wc = fftconvolve(cmd.astype(float),h)
            length = end+len(wc)+4800
            offset = (ri*9731+bi*31337)%len(noise)
            background = noise[(offset+np.arange(length))%len(noise)]
            noise_rms = np.sqrt(np.mean(background[end-len(original):end]**2))
            bg_gain = reference_rms/(noise_rms*10**(9/20))
            common = bg_gain*background
            common[end:end+len(wc)] += wc
            gain = min(1.,28000/(np.abs(common).max()+peak))
            s = dict(room=location,room_id=number,rir_sha256=rh,kind=kind,noise_path=path,noise_sha256=nh,
                     offset=offset, noise_gain=float(bg_gain), mic_gain=float(gain),
                     voice=voice,command=command,command_pcm_sha256=ch.digest(cmd),command_at=end,
                     baseline_wet_rms=float(reference_rms),samples=length)
            control=room.quantize(common*gain)
            for ppm in PPM:
                at = end - len(markers["v5"][ppm])
                for level in LEVELS:
                    signals={}
                    for name, waves in wet.items():
                        y = common.copy()
                        y[at:at+len(waves[ppm])] += 10**(level/20)*waves[ppm]
                        assert np.max(np.abs(y*gain))<=28001
                        signals[name] = room.quantize(y*gain)
                    yield dict(**s,start=at,player_ppm=ppm,level_db=level), signals, control


def outcome(detector,native,signal,s):
    route = fit.fixed.ch.routes(detector,native,signal)
    clocks = [gap.study.voices.timing.block(t) for t in route["merged"]]
    opened = clocks[0] if clocks else None
    return dict(**route, opened=opened, timely=opened is not None and s["start"]<=opened<s["command_at"],
                first_ms_before_command=(s["command_at"]-opened)/16 if opened is not None else None)


def main():
    native = room.rooms.c161_binding()
    assert ch.sha(room.rooms.MODEL) == room.rooms.MODEL_SHA
    rows,controls=[],{}
    total=defaultdict(lambda:defaultdict(Counter))
    by_rate=defaultdict(lambda:defaultdict(Counter))
    with TemporaryDirectory() as tmp:
        detector=fit.fixed.ch.binding(Path(tmp)/"v5.so",coded="v5")
        for s,signals,control in scenes():
            key=ch.digest(control)
            if key not in controls:
                controls[key]=dict(room=s["room"],kind=s["kind"],result=outcome(detector,native,control,s))
            outcomes={name:outcome(detector,native,w,s) for name,w in signals.items()}
            for group,address in ((total,str(s["level_db"])),(by_rate,str(s["player_ppm"]))):
                for name,r in outcomes.items():
                    group[address][name].update(trials=1,timely=r["timely"],opened=r["opened"] is not None,
                                              fixed=bool(r["fixed"]),c161=bool(r["c161"]))
                before,after=outcomes["v5"]["timely"],outcomes["trained"]["timely"]
                group[address]["paired"].update(gained=after and not before,lost=before and not after)
            rows.append(dict(source=s,control_sha256=key,routes=outcomes))
            if len(rows)%48==0: print(len(rows),flush=True)
    assert len(rows)==len(fit.EVAL)*2*len(LEVELS)*len(PPM) and len(controls)==len(fit.EVAL)*2
    ch.save(OUT,dict(records=rows,controls=controls,by_level=total,by_rate=by_rate,
        files={p.name:ch.sha(p) for p in (fit.FIT,fit.WAV,fit.fixed.CUE,room.rooms.MODEL,Path(__file__))},
        policy="Original and candidate at each identical scene clock, same frozen C161 and fixed matcher. "
        "14 development RIRs disjoint from this fit; established exposed source data. Nominal full-cue wet "
        "RMS 9 dB above the local background. Only cue level changes; mic gain constant across rates and levels "
        "in a room/background cell. Source clock moves only the emitted wake audio while commands keep their "
        "own clock. PCM is a digital simulation, absolute SPL and physical device timing unmeasured."))


if __name__ == "__main__":
    main()
