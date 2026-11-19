"""Stress fixed local playback with two reverberant paths, offsets, lower level and cold starts."""
import io
import re
import struct
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_speex_fixed_playback import AUDIT, BOT, LENGTH, OUT, RATE, ROOMS
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import boot, QEMU, transcripts

STEM = "4090-c161-speex-hold-rooms"
RIRS = RESULTS/"4090-mit-rir-audit.json"
ROOM_PAIRS = (("h001_", "h006_"), ("h003_", "h005_"))


def materials():
    assert sha(WAV)==read(AUDIT)["cue_sha256"]
    cue,rate=sf.read(WAV,dtype="int16")
    assert rate==RATE
    bank_path=RESULTS/f"{BOT}-pcm-v1.npz"
    assert sha(bank_path)==read(RESULTS/f"{BOT}-v1.json")["pcm_archive_sha256"]
    with np.load(bank_path) as bank:
        narration=bank["af_kore_digital"][40000:40000+LENGTH].astype(float)
    dry=RESULTS/"4090-c161-v5-household-rooms-dry.zip"
    assert sha(dry)==read(ROOMS)["waveform_archive_sha256"]
    with ZipFile(dry) as z:
        own, r1=sf.read(io.BytesIO(z.read("am_onyx-1-command.wav")),dtype="int16")
        other, r2=sf.read(io.BytesIO(z.read("af_heart-0-command.wav")),dtype="int16")
    assert r1==r2==RATE
    pad=np.zeros(320)
    return cue,narration,np.r_[cue,pad,own],np.r_[cue,pad,other]


def rir(prefix):
    entry,=[r for r in read(RIRS)["records"] if r["split"]=="training" and Path(r["path"]).name.startswith(prefix)]
    path=Path(".local/wakeword/mit-rirs")/entry["path"]
    assert sha(path)==entry["sha256"]
    h,rate=sf.read(path)
    assert rate==RATE
    return h/np.linalg.norm(h),dict(file=entry["path"],sha256=entry["sha256"])


def scenes():
    cue,narration,own_clip,ext_clip=materials()
    for room,pair in enumerate(ROOM_PAIRS):
        h,echo_src=rir(pair[0])
        g,near_src=rir(pair[1])
        # Use independent measured responses of the same room type for the two sources.
        for warm in (True, False):
            at=5*RATE if warm else 0
            ref=narration.copy()
            ref[at:at+len(own_clip)]=own_clip
            echo=.45*fftconvolve(ref,h)[:LENGTH]
            target=.45*fftconvolve(ext_clip,g)
            own_cue=fftconvolve(.45*cue,h)[:len(cue)]
            target_gain=np.linalg.norm(own_cue)/np.linalg.norm(target[:len(cue)])
            cases=[("self",0,0),("simultaneous",0,0)]
            if warm:
                cases.extend([("external_only",0,0),("reply_only",0,0)])
                cases.extend(("simultaneous",delay,level) for delay in (-1600,0,2400)
                             for level in (-6,0) if (delay,level)!=(0,0))
            for role,delay,db in cases:
                local=role in ("self","simultaneous")
                external=role in ("simultaneous","external_only")
                reference=ref if local else narration if role=="reply_only" else np.zeros_like(ref)
                mic=echo.copy() if local else .45*fftconvolve(reference,h)[:LENGTH]
                start=at+delay
                if external:
                    n=min(LENGTH-start,len(target))
                    mic[start:start+n]+=target[:n]*target_gain*10**(db/20)
                gain=min(1.,28000/max(abs(mic).max(),abs(reference).max(),1.))
                mic,reference=(np.rint(s*gain).astype("<i2") for s in (mic,reference))
                begin,end=(at,at+len(cue)) if local else (0,0)
                if local:
                    assert np.array_equal(reference[begin:end],np.rint(cue*gain).astype("<i2"))
                key=f"{room}-{int(warm)}-{role}-{delay}-{db}"
                # Include up to 100 ms of reverberant command tail in the capture.
                target_end=min(LENGTH,start+len(ext_clip)+1600)
                yield key,mic,reference,dict(room=room,warm=warm,role=role,offset=delay,external_db=db,
                      own_rir=echo_src,external_rir=near_src,external_gain_at_zero_db=target_gain,scale=gain,
                      known_playback=[begin,end],external_start=start if external else None,
                      command_end=target_end if external else None,mic_sha256=digest(mic),reference_sha256=digest(reference))


def score_external(reader, pcm, hits, meta):
    start = meta["external_start"]
    if start is None:
        return {}
    expected = start + 17588
    matched = [t for t in hits if abs(t - expected) <= 1600]
    result = dict(external_window_hits=matched)
    if matched:
        first = max(0, (matched[0] + 159) // 160 * 160 - 32000)
        words = transcripts(reader, pcm[first:meta["command_end"]])
        want = normalized("Turn off the desk lamp.")
        result.update(first_sample=first, transcript=words,
                      exact_suffix=normalized(words["text"])[-len(want):] == want)
    return result


def main():
    inputs=list(scenes())
    root=OUT/"two-paths"
    root.mkdir(exist_ok=True)
    plan=RESULTS/f"{STEM}-plan.json"
    config=dict(rooms=ROOM_PAIRS,source_sha256={f.name:sha(f) for f in (WAV,ROOMS,RIRS)},
                records=[dict(key=k,**m) for k,_,_,m in inputs])
    if plan.exists():
        assert read(plan)["records"]==config["records"]
    else:
        save(plan,config)
    reader,identity=load_reader()
    rows=[]
    archive=RESULTS/f"{STEM}-uart.zip"
    if archive.exists() or (RESULTS/f"{STEM}.json").exists():
        raise FileExistsError(STEM)
    with ZipFile(root/"uart.zip","w",ZIP_DEFLATED) as logs:
        for key,mic,ref,meta in inputs:
            row=dict(key=key,arms={})
            for arm in ("baseline","hold"):
                build=OUT/("build-hold" if arm=="hold" else "build-cue")
                header=struct.pack("<II",0x35434541,len(mic))
                payload=header+mic.tobytes()+ref.tobytes()
                if arm=="hold":
                    payload+=struct.pack("<II",*meta["known_playback"])
                log=root/f"{key}-{arm}.log"
                boot(build,log,root/"flash.bin",b"SPEEX PASS",(2<<20,payload))
                pcm,costs,heap=parse(log,len(mic))
                hits=list(map(int,re.findall(r"SPEEX_CUE (\d+)\n",log.read_text())))
                r=dict(pcm_sha256=digest(pcm),uart_sha256=sha(log),hits=hits,
                       median_us=float(np.median(costs)),max_us=max(costs),over_10ms=sum(v>10000 for v in costs))
                r.update(score_external(reader, pcm, hits, meta))
                row["arms"][arm]=r
                logs.write(log,log.name)
            if meta["known_playback"]==[0,0]:
                assert row["arms"]["baseline"]["pcm_sha256"]==row["arms"]["hold"]["pcm_sha256"]
            rows.append(row)
            print(key,[v["hits"] for v in row["arms"].values()],flush=True)
    (root/"uart.zip").replace(archive)
    save(RESULTS/f"{STEM}.json",dict(plan_sha256=sha(plan),asr=identity,qemu_sha256=sha(QEMU),
         elfs={n:sha(OUT/n/"nyx_speex_echo.elf") for n in ("build-hold","build-cue")},records=rows,uart_archive_sha256=sha(archive)))


if __name__=="__main__":
    main()
