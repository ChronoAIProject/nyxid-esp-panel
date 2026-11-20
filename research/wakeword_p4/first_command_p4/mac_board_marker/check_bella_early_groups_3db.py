"""Replay paired interruption shifts and isolate the known 079 miss."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf
import torch

import bella_early_groups_3db as candidate
import bella_third_triplet_player_level as level
import package_third_triplet_stereo as handoff

P = candidate.P
source = candidate.triple.experiment.source
fixed = source.fixed
WORK = candidate.WORK
OUT = candidate.FIT.with_name("4090-bella-early-groups-3db-shifts.json")
GAIN = 10**(-6/20)
PLAIN = OUT.with_name("c161-eight-voice-willow-comparison.zip")
MARKED = OUT.with_name("c161-eight-voice-marker-gains.zip")


def member_wave(archive, name, checksum, dtype):
    raw = archive.read(name)
    assert sha256(raw).hexdigest() == checksum
    pcm, rate = sf.read(BytesIO(raw), dtype=dtype)
    assert rate == 16000
    return pcm


def carrier():
    with ZipFile(PLAIN) as archive:
        entry, = (r for r in json.loads(archive.read("manifest.json"))["records"]
                  if (r["batch"], r["track"], r["role"]) == ("second", 0, "nyx_plain"))
        pcm = member_wave(archive, entry["file"], entry["wav_sha256"], "int16")
    with ZipFile(MARKED) as archive:
        record, = (r for r in json.loads(archive.read("manifest.json"))["records"]
                   if (r["batch"], r["index"]) == ("second", 0))
        wake = record["wake_only"]
        marked = member_wave(archive, wake["file"], wake["sha256"], "int16")
    assert record["voice"] == entry["voice"] == "af_bella"
    prefix = np.rint(pcm[32000:entry["command_at"]]*record["prefix_scale"])
    assert prefix.shape == marked.shape == (23200,)
    keep = np.ones(len(prefix), dtype=bool)
    for start in (16160,18720,20640):
        keep[start:start+1280] = False
    np.testing.assert_array_equal(prefix[keep], marked[keep])
    single, _ = sf.read(candidate.SINGLE, dtype="int16")
    np.testing.assert_array_equal(single, np.rint(marked*10**.25))
    return np.rint(prefix*10**.25)


def shift(signal, samples):
    if samples == 0:
        return signal
    result = np.zeros_like(signal)
    if samples > 0:
        result[samples:] = signal[:-samples]
        assert not signal[-samples:].any()
    else:
        result[:samples] = signal[-samples:]
        assert not signal[:-samples].any()
    return result


def inputs(mode):
    old, sr = sf.read(candidate.OLD, dtype="int16")
    new, rr = sf.read(candidate.WAV, dtype="int16")
    assert sr == rr == 16000 and P.sha(candidate.WAV) == P.read(candidate.FIT)["voices"][candidate.NAME]
    waves = {"old":old, "new":new, "erased":np.zeros_like(old)}
    if mode == "diagnosis":
        voice = carrier()
        waves.update(voice=voice, markers=old-voice)
    parent = P.read(level.OUT)
    assert P.sha(level.PLAN) == parent["plan_sha256"]
    rows = {r["label"]:r for r in parent["records"]}
    plan = P.read(level.angles.PLAN)
    scale = P.read(level.study.PLAN)["paths"][0]["common_scale"]
    rng = np.random.Generator(np.random.PCG64(631820))
    receipt = P.read(handoff.OUT)
    assert P.sha(handoff.PACK) == receipt["archive_sha256"]
    with ZipFile(handoff.PACK) as archive:
        for angle in level.angles.GIT_BLOBS:
            mat = level.angles.WORK/f"angle_{angle}.0.mat"
            assert P.sha(mat) == plan["files"][str(mat)]
            ir = loadmat(mat)["brir"]
            front, back = (resample_poly(ir[:,1,ch],1,3)/scale for ch in (2,8))
            for side, target, other in (("front",front,back),("back",back,front)):
                arrival = int(np.abs(target[:800]).argmax())
                for scene in (77,79,90):
                    key = f"{angle}-{side}-{scene:03}"
                    row = rows[key+"-three-triplets-target-minus6"]
                    erased = rows[key+"-absent-target-minus6"]
                    noise = rng.normal(0,12,row["samples"])
                    if mode == "diagnosis" and (angle,side,scene) != (210,"front",79):
                        continue
                    name = f"{scene:03}/three-triplets.wav"
                    stereo = member_wave(archive,name,receipt["files"][name],"float64")
                    end = row["end"]-arrival
                    address = slice(end-len(old),end)
                    np.testing.assert_array_equal(stereo[address,0]*131072, old)
                    gaps = (0, None) if mode == "diagnosis" else (-160,-80,0,80,160)
                    for offset in gaps:
                        right = (np.zeros_like(noise) if offset is None else
                                 fftconvolve(32768*shift(stereo[:,1],offset*16),other))
                        for role, wave in waves.items():
                            left = 32768*stereo[:,0]
                            left[address] = wave.astype(float)/4
                            target_mic = GAIN*fftconvolve(left,target)
                            raw = np.rint(target_mic + (right + noise))
                            mic = raw.clip(-32768,32767).astype("<i2")
                            expected = None
                            if offset == 0 and role in ("old","erased"):
                                expected = row if role == "old" else erased
                                assert P.digest(mic) == expected["pcm_sha256"]
                            yield row, role, offset, mic, target_mic, right, expected, int(np.sum(mic!=raw))


def main(mode):
    torch.set_num_threads(1)
    trace = source.trace
    fir = trace.coefficients("kCueBandFirV5")
    patterns = torch.tensor(trace.coefficients("kCueBandPatternsV5").reshape(3,320),
                            dtype=torch.float64,device="cuda")
    assert "4090" in torch.cuda.get_device_name()
    build = P.read(source.receiver.WORK/"build.json")
    flash = bytearray((source.receiver.WORK/"guest.flash").read_bytes())
    assert build == P.read(level.OUT)["build"] and fixed.digest(flash) == build["flash_sha256"]
    assert P.sha(fixed.QEMU) == build["qemu_sha256"] and P.sha(fixed.MODEL) == build["model_sha256"]
    folder = WORK/mode
    cache = WORK/build["flash_sha256"]
    cache.mkdir(exist_ok=True,parents=True)
    folder.mkdir(exist_ok=True)
    output = OUT if mode == "shifts" else OUT.with_stem(OUT.stem.replace("shifts","diagnosis"))
    rows, files = [], {}
    with TemporaryDirectory(dir=folder) as tmp:
        work = Path(tmp)
        with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as logs:
            for prior, role, offset, mic, target, talk, expected, clipped in inputs(mode):
                ref = np.zeros_like(mic)
                assert P.digest(ref) == prior["ref_sha256"]
                label = f"{prior['angle']}-{prior['direction']}-{prior['scene']:03}-{role}-{offset}"
                block = fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                active_cache = level.WORK/build["flash_sha256"] if expected else cache
                for attempt in range(2):
                    try:
                        native, log, faults = fixed.boot(work,flash,active_cache,block,1)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                events = native[0]["events"]
                if expected:
                    assert events == expected["nyx"]
                for path in (log,*faults):
                    member = label + ("/wake.log" if path==log else "/"+path.name)
                    files[member] = P.sha(path)
                    logs.write(path,member)
                start = prior["start"] if role != "erased" else None
                opened = next(iter(events["O"]), None)
                row = {k:prior[k] for k in ("angle","direction","scene","command","end","ref_sha256","reference_path")}
                row.update(label=label,start=start,offset_ms=offset,role=role,events=events,clipped=clipped,
                           pcm_sha256=P.digest(mic),input_sha256=fixed.digest(block),parent=prior["label"],
                           timely=opened is not None and start is not None and start<=opened<prior["end"])
                if mode == "diagnosis":
                    groups = {}
                    for name, advance in (("early",12800),("middle",6400),("late",0)):
                        starts = [prior["end"]-23200+s-advance for s in (16160,18720,20640)]
                        groups[name] = source.measure_triplet(mic,target,talk,starts,fir,patterns)
                    chains = [g["chain_sample"] for g in groups.values() if g["chain_sample"] is not None]
                    assert events["C"] == ([min(chains)] if chains else []), (label, events, chains)
                    row["groups"] = groups
                if (prior["angle"],prior["direction"],prior["scene"],offset,role) == (210,"front",79,0,"new"):
                    path = folder/(label+".wav")
                    sf.write(path,mic,16000,subtype="PCM_16")
                    row.update(path=str(path),sha256=P.sha(path))
                rows.append(row)
                print(label,events,flush=True)
        assert len(rows) == len({r["label"] for r in rows}) == (10 if mode=="diagnosis" else 180)
        deps = (Path(__file__).resolve().relative_to(Path.cwd()),candidate.FIT,candidate.PROTO,
                handoff.OUT,level.OUT,level.angles.PLAN,level.study.PLAN,trace.COEFF,PLAIN,MARKED)
        P.save(work/output.name,dict(records=rows,build=build,files=files,archive_sha256=P.sha(work/"uart.zip"),
                                     gpu=torch.cuda.get_device_name(),sources={str(p):P.sha(p) for p in deps}))
        (work/"uart.zip").replace(output.with_suffix(".zip"))
        (work/output.name).replace(output)


if __name__ == "__main__":
    mode, = sys.argv[1:]
    assert mode in ("shifts","diagnosis")
    main(mode)
