"""Align the final active wake frame with the first active command frame in four new voice pairings."""
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import voice_cues_willow as matched
from check_c161_command_playback_level import limited, recognition
from check_c161_v5_code_gain import routes
from coded_command_uplink import write_member
from transition_inputs import commands, read_noise, read_room
from voice_cues import RESULTS, binding, c161_binding, codes, digest, load_reader, read, save, sha, variants

STEM = "4090-c161-coded-close-join"
PLAN = RESULTS / f"{STEM}-plan.json"
DRY = RESULTS / f"{STEM}-dry.zip"
RUNS = Path(".local/wakeword/c161-coded-close-join")
CASE = Path(__file__).with_name("coded_join_case.py").resolve()
VOICES = ("af_bella", "am_eric", "bf_lily", "bm_fable")
NAMES = ("Hey Nyx.", "Hey Willow.", "Hey Nick.")
TEXTS = ("Close the upstairs curtains.", "Switch off the garden lights.")
ROLES = ("ordinary", "coded", "willow", "nick", "silence")
COMMAND_AT = 57600


def active(pcm):
    power = np.mean(np.pad(pcm.astype(float)**2, (0,-len(pcm)%320)).reshape(-1,320), axis=1)
    frames = np.flatnonzero(power >= power.max()*.001)
    return 320*int(frames[0]), min(len(pcm),320*(int(frames[-1])+1))


def dry_bank():
    if not DRY.exists():
        labels = [(v,t) for v in VOICES for t in (*NAMES,*TEXTS)]
        speech, files = commands(labels)
        with ZipFile(DRY.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
            rows = []
            for i,((v,t),pcm) in enumerate(zip(labels,speech,strict=True)):
                rows.append(dict(voice=v,text=t,**write_member(z,f"{i:02}.wav",pcm)))
            z.writestr("manifest.json",json.dumps(dict(files=files,records=rows),indent=2)+"\n")
        DRY.with_suffix(".tmp").replace(DRY)
    bank = {}
    with ZipFile(DRY) as z:
        manifest = json.loads(z.read("manifest.json"))
        for r in manifest["records"]:
            pcm,rate = sf.read(io.BytesIO(z.read(r["file"])),dtype="int16")
            assert rate == 16000 and digest(pcm) == r["pcm_sha256"]
            bank[r["voice"],r["text"]] = pcm
    return bank, manifest["files"]


def prepare():
    RUNS.mkdir(exist_ok=True)
    bank, sources = dry_bank()
    previous = read(matched.PLAN)
    files = {p:sha(Path(p)) for p in previous["runtime"].values()}
    assert all(previous["files"][p] == h for p,h in files.items())
    files.update(sources)
    files.update({str(p):sha(p) for p in (CASE, Path(__file__).resolve(), DRY, matched.official.official.SOURCE,
                                           Path(matched.official.official.__file__))})
    rows = []
    for rir, noise_kind in (("h012_","air_conditioner"), ("h004_","music")):
        h,room,room_sha = read_room(rir)
        noise,_,noise_sha = read_noise(noise_kind)
        for i,v in enumerate(VOICES):
            nyx,willow,nick = (bank[v,n] for n in NAMES)
            pos, clips = variants(nyx,codes())
            ends = [active(pcm)[1] for pcm in (nyx, willow, nick)]
            assert pos+5760 == ends[0]
            cues = (nyx[:ends[0]], clips["coded"][:ends[0]], willow[:ends[1]], nick[:ends[2]], np.zeros(ends[0],np.int16))
            for j,t in enumerate(TEXTS):
                source = bank[v,t]
                begin,_ = active(source)
                # The playback file starts with the first active 20 ms frame of the command.
                # Scale the room noise against the unboosted command, then hold it constant across roles.
                wet_raw = fftconvolve(source[begin:].astype(float),h)
                cmd = limited(source)[begin:]
                wet = fftconvolve(cmd.astype(float),h)
                at = COMMAND_AT
                length = at+len(wet)+8000
                offset = (89+13*i+5*j)*16000 % (len(noise)-length+1)
                bg = noise[offset:offset+length]
                p = np.mean(np.pad(wet_raw**2,(0,-len(wet)%320)).reshape(-1,320),axis=1)
                mask = np.repeat(p>=p.max()*.001,320)[:len(wet)]
                gain = np.linalg.norm(wet_raw[mask])/np.linalg.norm(bg[at:at+len(wet)][mask])
                common = bg*gain
                common[at:at+len(wet)] += wet
                streams = []
                for cue in cues:
                    mic = common.copy()
                    filtered = fftconvolve(cue.astype(float),h)
                    start = at-len(cue)
                    mic[start:start+len(filtered)] += filtered
                    streams.append(mic)
                scale = min(1.,28000/max(np.max(abs(m)) for m in streams))
                for role,cue,mic in zip(ROLES,cues,streams,strict=True):
                    pcm = np.rint(scale*mic).astype("<i2")
                    file = RUNS/f"{len(rows):03}.wav"
                    sf.write(file,pcm,16000,subtype="PCM_16")
                    files[str(file)] = sha(file)
                    rows.append(dict(path=str(file), sha256=files[str(file)], pcm_sha256=digest(pcm), samples=len(pcm),
                        voice=v,carrier=t,target_id=role,start=at-len(cue),end=at,room=room,rir_sha256=room_sha,
                        noise_sha256=noise_sha,noise_offset=offset,noise_gain=gain,common_gain=scale,
                        command_trim_samples=begin,command_pcm_sha256=digest(cmd),cue_pcm_sha256=digest(cue),
                        command_snr_db=float(20*np.log10(np.linalg.norm(wet[mask])/np.linalg.norm(gain*bg[at:at+len(wet)][mask])))))
    assert len(rows)==80
    save(PLAN,dict(runtime=previous["runtime"], files=files, records=rows, threshold_requests=["DEFAULT"],
        policy="New generated test utterances from four previously admitted Kokoro training voices; consumed room/noise sources. For every role, the last active wake frame ends at 57600; the first active source command frame begins at 57600. Source commands use the frozen limiter. Same room, noise phase, gain, command and notification deadline within each pair."))


def score():
    native = c161_binding()
    reader, identity = load_reader()
    plan = read(PLAN)
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(RESULTS/f"{STEM}-uart.zip","x",ZIP_DEFLATED) as z:
        detector = binding(Path(tmp)/"full.so",coded="v5")
        for i,s in enumerate(plan["records"]):
            run = RUNS/"runs"/f"{i:02}"
            report = read(run/"report.json")
            assert report["model"]=="wn9_heywillow_tts" and report["total_prime_samples"]==81600
            assert report["actual_float32_bits"]==report["post_prime_getter_bits"]=="3f24dd2f" and report["stage"]==104
            assert report["execution_sha256"]==sha(PLAN) and report["source_pcm_sha256"]==s["pcm_sha256"]
            for name,field in (("raw-returns.bin","raw_returns_sha256"),("serial.log","serial_sha256")):
                assert sha(run/name)==report[field]
                z.write(run/name,f"{i:02}/{name}")
            z.write(run/"report.json",f"{i:02}/report.json")
            raw=np.fromfile(run/"raw-returns.bin",np.uint8)
            willow=(np.minimum((np.flatnonzero(raw==2)+1)*512,report["input_samples"])-49344).tolist()
            assert willow==[r["relative_sample"] for r in report["real_events"]] and not report["additional_priming_events"]
            pcm,rate=sf.read(s["path"],dtype="int16")
            assert rate==16000 and digest(pcm)==s["pcm_sha256"] and sha(Path(s["path"]))==s["sha256"]
            nyx=routes(detector,native,pcm)
            arms={}
            for label,ts in (("nyx",nyx["merged"]),("willow",willow)):
                item=dict(events=ts)
                if ts:
                    event=ts[0]
                    delivered=(event+255)//256*256 if label=="nyx" else event
                    item.update(first_in_cue=s["start"]<=event<s["end"],before_command=s["start"]<=delivered<s["end"],
                                first_delivery=delivered,offset_ms=(delivered-s["end"])/16)
                    if s["target_id"] in ("ordinary","coded","willow") and s["start"]<=event<len(pcm):
                        begin=max(0,delivered//160*160-32000)
                        item.update(begin=begin,capture=recognition(reader,pcm[begin:],s["carrier"]))
                arms[label]=item
            rows.append(dict(index=i,pcm_sha256=s["pcm_sha256"],voice=s["voice"],carrier=s["carrier"],room=s["room"],
                             role=s["target_id"],c161=nyx["c161"],fixed=nyx["fixed"],arms=arms))
    save(RESULTS/f"{STEM}.json",dict(plan_sha256=sha(PLAN),model_sha256=matched.MODEL_SHA,asr_files=identity,
         uart_sha256=sha(RESULTS/f"{STEM}-uart.zip"),records=rows))


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="willow":
        matched.official.PLAN=PLAN
        matched.official.OUT=RUNS/"runs"
        matched.official.__file__=str(CASE)
        matched.official.willow()
    else:
        {"prepare":prepare,"score":score}[action]()
