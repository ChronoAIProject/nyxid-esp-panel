"""Calibrate the auxiliary classifier's temporal averaging on separate synthetic commands."""
from collections import defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.signal import fftconvolve

import replay_post120_overlap as replay
from check_additional_connected_voices import onset

ch, fit = replay.ch, replay.score.fit
ROOT = ch.RESULTS
STEM = "4090-c161-unmarked-dwell"
OUT = ROOT / (STEM+"-calibration.json")
WAVES = ROOT / (STEM+"-training-waves.npz")
VOICES = ("af_alloy","af_aoede","af_heart","af_jessica","am_adam","am_fenrir","am_michael","am_santa")
COMMANDS = ("set the desk lamp to blue.", "read the next delivery time.", "pause playback in the study.")
ADDRESSES = ("Hey Nyx","Hey Nicks","Hey Nick","Hey Nicole","Hey Max","Hey Willow","What's next","It takes")
KINDS = ("target","target","name","name","name","name","everyday","everyday")
BOUNDARY = 48000
WIDTHS = (1,3,5)


def syntheses():
    with patch.multiple(onset, VOICES=VOICES, CARRIERS=COMMANDS, ADDRESSES=ADDRESSES, LABELS=KINDS):
        for s, pcm in onset.synthesize():
            words = list(re.finditer(r"[^ ,.!?]+",s["phones"]))
            end_pos = words[2].start()
            while s["phones"][end_pos] in "ˈˌ":
                end_pos += 1
            edges = np.cumsum(s["duration"],dtype=int)*400
            end = int(edges[end_pos])
            offset = BOUNDARY-end
            assert offset >= 0
            wave = np.pad(pcm,(offset,16000))
            yield s, pcm, wave, offset+int(edges[words[0].start()]), end


def sums(codes, width):
    return np.convolve(codes.astype(np.int32), np.ones(width,np.int32), "valid")


def detect(codes, clock, width, threshold):
    levels = sums(codes,width)
    candidate = np.flatnonzero((levels>=threshold) | (levels*2<threshold))
    last, armed, found = -1000, True, []
    for i in candidate:
        if i-last < 67:
            continue
        if levels[i]*2 < threshold:
            armed = True
        elif armed:
            last,armed = i,False
            t = int(clock[i+width-1])
            if t > 0:
                found.append(t)
    return found


def calibration(rows, arrays):
    negative = defaultdict(list)
    for r in rows:
        if r["kind"]!="target":
            negative[r["kind"]].append(r)
    original = {}
    for kind,items in negative.items():
        original[kind] = sum(bool(r["parent_events"]) for r in items)
    policies = {}
    for width in WIDTHS:
        # Require every negative stratum to stay within the parent's clip-level false count.
        peaks = {}
        prime_peak = max(r["peaks"][str(width)]["prime"] for r in rows)
        cuts = [max(128*width,prime_peak+1)]
        for kind,items in negative.items():
            peaks[kind] = sorted((r["peaks"][str(width)]["whole"] for r in items),reverse=True)
            if original[kind] < len(items):
                cuts.append(peaks[kind][original[kind]]+1)
        threshold = max(cuts)
        hits,early,voices,counts = 0,0,set(),defaultdict(int)
        for r in rows:
            ts = detect(arrays[r["key"]+"_candidate"],arrays[r["key"]+"_clock"],width,threshold)
            t = (ts[0]+255)//256*256 if ts else None
            if r["kind"]=="target":
                good = t is not None and r["wake_start"]<=t<=BOUNDARY+2400
                if good:
                    voices.add(r["voice"])
                hits += int(good)
                early += int(good and t<BOUNDARY)
            else:
                counts[r["kind"]] += int(bool(ts))
        assert all(counts[k]<=original[k] for k in negative)
        policies[str(width)] = dict(width=width,threshold=threshold,target_by_150ms=hits,target_before=early,
                                    voices=len(voices),negative=dict(counts))
    selected = max(WIDTHS,key=lambda w:(policies[str(w)]["voices"],policies[str(w)]["target_by_150ms"],
                    policies[str(w)]["target_before"],-sum(policies[str(w)]["negative"].values()),w))
    return original,policies,str(selected)


def main():
    models, hashes = replay.models()
    previous = ch.read(replay.OUT)
    exposed = {r["source"]["voice"] for r in previous["records"] if "cohort" in r}
    assert set(VOICES).isdisjoint(exposed) and set(COMMANDS).isdisjoint(
        r["source"]["command"] for r in previous["records"] if "cohort" in r)
    native = fit.prior.letters.native_frontend()
    beds, files, rooms, demos = fit.acoustics()
    rng = np.random.default_rng(83716)
    rows, arrays, waves, sources = [], {}, {}, []
    for i,(s,pcm,wave,start,end) in enumerate(syntheses()):
        if i%len(ADDRESSES)==0:
            hx,hp = rng.choice(len(rooms),2,replace=False)
            bed_index = int(rng.integers(len(beds)))
            bed = beds[bed_index]
            offset = int(rng.integers(len(bed)))
            panel_at = BOUNDARY+int(rng.integers(-23000,-12000))
            db = (0,9)[(i//len(ADDRESSES))%2]
            probe = fftconvolve(wave.astype(float),rooms[hx][1])
            speech = probe[start:BOUNDARY]
            active = np.abs(speech)>max(1,np.abs(speech).max()*.035)
            rms = np.sqrt(np.mean(speech[active]**2))
            music_gain = rms*10**(-db/20)/np.sqrt(np.mean(bed**2))
            demo = demos[hp]
            active = np.abs(demo)>max(1,np.abs(demo).max()*.035)
            panel_gain = rms/np.sqrt(np.mean(demo[active]**2))
        song = dict(external_room=rooms[hx][0],panel_room=rooms[hp][0],music_sha256=files[bed_index]["sha256"],
                    offset=offset,panel_at=panel_at,snr_setting_db=db,panel_gain=float(panel_gain),music_gain=float(music_gain))
        key = f"wav{i:03}"
        waves[key] = pcm
        sources.append(dict(s,key=key,samples=len(pcm),pcm_sha256=ch.digest(pcm),command_at=end,scene=song))
        variants = [(s["label"],wave)]
        if s["label"]=="target":
            absent = wave.copy()
            absent[:BOUNDARY] = 0
            assert np.array_equal(absent[BOUNDARY:],wave[BOUNDARY:])
            variants.append(("no_address",absent))
        wet = [fftconvolve(x.astype(float),rooms[hx][1]) for _,x in variants]
        size = max(len(wet[0]),panel_at+len(demo))
        noise = bed[(offset+np.arange(size))%len(bed)]*music_gain
        noise[panel_at:panel_at+len(demo)] += demo*panel_gain
        mixes = [noise+np.pad(x,(0,size-len(x))) for x in wet]
        gain = min(1.,30000/max(np.abs(x).max() for x in mixes))
        for (kind,dry),mix in zip(variants,mixes,strict=True):
            for condition,mic in (("dry",dry),("mixed",np.rint(mix*gain).astype("<i2"))):
                _,bands,clock = fit.prior.base.native_inputs(native,mic)
                k = f"s{len(rows):03}"
                arrays[k+"_clock"] = clock
                peaks = {}
                for stream,name in (("parent","original"),("candidate","command")):
                    trace = fit.prior.base.infer(models[name],bands)
                    arrays[k+"_"+stream] = trace
                    if stream=="parent":
                        parent_events = detect(trace,clock,5,1175)
                        assert parent_events == [t for t in replay.score.times(replay.score.decode(trace,clock)) if t>0]
                    else:
                        for width in WIDTHS:
                            v=sums(trace,width)
                            real=clock[width-1:]>0
                            peaks[str(width)] = dict(prime=int(v[~real].max()),whole=int(v[real].max()))
                rows.append(dict(key=k,source_key=key,kind=kind,condition=condition,voice=s["voice"],wake_start=start,
                                 mic_sha256=ch.digest(mic),gain=gain if condition=="mixed" else 1.,
                                 parent_events=parent_events,peaks=peaks))
    assert len(sources)==192 and len(rows)==480
    budgets,policies,selected = calibration(rows,arrays)
    with TemporaryDirectory(dir=ROOT) as tmp:
        work=Path(tmp)
        for path,data in ((WAVES,waves),(OUT.with_suffix(".npz"),arrays)):
            np.savez_compressed(work/path.name,**data)
            with np.load(work/path.name,allow_pickle=False) as check:
                for k,v in data.items():
                    np.testing.assert_array_equal(check[k],v)
        ch.save(work/OUT.name,dict(records=rows,sources=sources,models=hashes,original_negative=budgets,policies=policies,
                selected=selected,waves_sha256=ch.sha(work/WAVES.name),codes_sha256=ch.sha(work/OUT.with_suffix(".npz").name),
                script_sha256=ch.sha(Path(__file__)),synthesis_sha256=ch.sha(Path(onset.__file__)),
                policy="Training-pool synthetic names, sentences and removed-address counterparts. Test 1, 3 and 5 "
                       "ticks on the frozen post120 command auxiliary. Each negative group capped at original proposer's "
                       "number of triggered clips on these same recordings. Select on these training inputs only."))
        for path in (WAVES,OUT.with_suffix(".npz"),OUT):
            (work/path.name).replace(path)


if __name__=="__main__":
    main()
