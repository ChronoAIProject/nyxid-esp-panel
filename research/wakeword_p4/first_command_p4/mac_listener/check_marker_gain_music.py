"""Replay frozen single-triple cues across previously used rooms and 11 additional music recordings."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

import fit_single_marker_gains as fit

grid, ch, prior = fit.grid, fit.ch, fit.prior
OUT = ch.RESULTS / (fit.STEM + "-rooms.json")
ROOMS = (*grid.fit.EVAL, *prior.repeat.source.EXTRA)


def tracks():
    original = grid.fit.original()[1]
    outputs = {"v5":original}
    for name, wav, report in (("boosted",prior.WAV,prior.OUT), ("markers",fit.WAV,fit.FIT)):
        pcm, rate = sf.read(wav,dtype="int16")
        assert rate == 16000 and ch.sha(wav) == ch.read(report)["wav_sha256"] and len(pcm) == len(original)
        outputs[name] = pcm
    return outputs


def more_music(dry):
    songs = ch.read(fit.DEV)["records"][1:]
    assert len(songs) == 11
    training = ch.read(fit.FIT)["music"]
    assert {s["sha256"] for s in songs}.isdisjoint(r["sha256"] for r in training)
    fitted_rooms = {r["sha256"] for r in ch.read(grid.fit.FIT)["training_rooms"]}
    bank = grid.room.rooms.source_bank()
    commands = sorted({(s["voice"],s["carrier"]) for s in ch.read(grid.room.rooms.PLAN)["records"]})
    warped = {ppm:{name:grid.drift.project(wav,Fraction(1000000,1000000+ppm)) for name,wav in dry.items()}
              for ppm in grid.PPM}
    for r, room_id in enumerate(ROOMS):
        h, path, rh = grid.room.rooms.read_room(f"h{room_id:03}_")
        assert rh not in fitted_rooms
        reverbs = {ppm:{name:fftconvolve(wav.astype(float),h) for name,wav in waves.items()} for ppm,waves in warped.items()}
        base = reverbs[0]["v5"][:len(dry["v5"])]
        for j, song in enumerate(songs):
            voice,command = commands[(r+3*j)%len(commands)]
            cmd = bank[voice,command]
            wc = fftconvolve(cmd.astype(float),h)
            ppm = grid.PPM[(r+j)%3]
            end = 51200
            start = end-len(warped[ppm]["v5"])
            length = end+len(wc)+4800
            song_path = Path(song["path"])
            if r == 0:
                assert ch.sha(song_path) == song["sha256"]
            with sf.SoundFile(song_path) as f:
                assert f.samplerate == 16000 and f.channels == 1
                offset = (2654435761*(r+1)+2246822519*(j+1))%(len(f)-length+1)
                f.seek(offset)
                bg = f.read(length)
            # Keep the acoustic reference level anchored to the original unwarped v5 cue.
            noise_gain = np.linalg.norm(base)/(np.linalg.norm(bg[end-len(base):end])*10**(9/20))
            mixed = bg*noise_gain
            mixed[end:end+len(wc)] += wc
            peak = max(np.abs(w).max() for w in reverbs[ppm].values())
            gain = min(1.,28000/(np.abs(mixed).max()+peak))
            control = grid.room.quantize(mixed*gain)
            for level in (-12,-18):
                signals = {}
                for name,wet in reverbs[ppm].items():
                    wave = mixed.copy()
                    wave[start:start+len(wet)] += wet*10**(level/20)
                    assert np.abs(wave*gain).max() <= 32767
                    signals[name] = grid.room.quantize(wave*gain)
                source = dict(room=path,room_id=room_id,rir_sha256=rh,track=song["source_path"],
                              noise_sha256=song["sha256"],offset=offset,noise_gain=float(noise_gain),mic_gain=float(gain),
                              command_at=end,start=start,samples=length,player_ppm=ppm,level_db=level,voice=voice,
                              command=command,command_pcm_sha256=ch.digest(cmd))
                yield source, signals, control


def established(dry):
    for ids in (grid.fit.EVAL,prior.repeat.source.EXTRA):
        for s,signals,control in grid.scenes(ids,{n:w for n,w in dry.items() if n!="v5"}):
            yield s,{n:signals[n] for n in dry},control


def main():
    dry = tracks()
    original = ch.read(prior.repeat.OUT)
    boost = ch.read(prior.level.OUT)
    model = grid.room.rooms.MODEL
    assert ch.sha(model) == grid.room.rooms.MODEL_SHA
    native = grid.room.rooms.c161_binding()
    summary = defaultdict(lambda:defaultdict(Counter))
    rows,controls = [],{}
    with TemporaryDirectory() as tmp:
        full = ch.binding(Path(tmp)/"full.so",coded="v5")
        pair = ch.binding(Path(tmp)/"pair.so",coded="aligned")
        for group,inputs in (("established",established(dry)),("other_music",more_music(dry))):
            for i,(s,signals,control) in enumerate(inputs):
                key = ch.digest(control)
                if key not in controls:
                    ctrl = grid.outcome(full,native,control,s)
                    controls[key] = dict(group=group,opened=ctrl["opened"],c161=ctrl["c161"],fixed=ctrl["fixed"])
                results = {}
                for name,x in signals.items():
                    result = grid.outcome(full,native,x,s)
                    if group == "established":
                        assert s == original["records"][i]["source"] and key == original["records"][i]["control_sha256"]
                        if name=="v5":
                            old = original["records"][i]["routes"]["v5"]
                            assert all(old[k] == v for k,v in result.items())
                        if name=="boosted":
                            assert boost["records"][i]["routes"]["boosted_v5"]["external"] == result
                    ref = prior.level.guard.refs(s,name,dry)
                    guard = grid.drift.guarded(result,full,pair,ref)
                    results[name] = dict(pcm_sha256=result["pcm_sha256"],reference_sha256=guard["reference_sha256"],
                                         c161=result["c161"],fixed=result["fixed"],opened=result["opened"],timely=result["timely"],
                                         self_opened=guard["opened"],reference_events=guard["reference_events"])
                    base = results["v5"]
                    summary[group][name].update(trials=1,timely=result["timely"],opened=result["opened"] is not None,
                                                self_open=bool(guard["opened"]),gained=result["timely"] and not base["timely"],
                                                lost=base["timely"] and not result["timely"])
                rows.append(dict(group=group,index=i,source=s,control_sha256=key,routes=results))
                if (i+1)%112 == 0: print(group,i+1,flush=True)
    assert len(rows)==672+616 and len(controls)==56+308
    ch.save(OUT,dict(records=rows,controls=controls,summary=summary,model_sha256=ch.sha(model),
                     sources={p.name:ch.sha(p) for p in (fit.FIT,fit.WAV,fit.DEV,prior.OUT,prior.repeat.OUT,prior.level.OUT,Path(__file__))},
                     policy="Frozen C161 and v5 combiner, unchanged original reference veto. Reproduce all 672 "
                     "earlier v5 and gain-only observations. Additional 11 previously exposed music recordings "
                     "by 28 rooms, two cue levels per excerpt. Balanced rotating clock errors, fixed cue energy "
                     "reference, shared command and microphone level. Ignore neither early false events nor late wakes. "
                     "No new background-duration or population-generalization claim."))


if __name__ == "__main__":
    main()
