"""Render new first-order rectangular rooms and replay two frozen generated calls."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path("research/wakeword_p4")
STEM = "4090-bella-prospective-rooms"
PLAN = ROOT/"results"/(STEM+"-plan.json")
OUT = PLAN.with_name(STEM+".json")
WORK = Path(".local/wakeword/bella-prospective-rooms")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
SEED = 318491


def image_response(dimensions, mic, source, reflections):
    import numpy as np
    images = [(source,1.)]
    for axis in range(3):
        for side in range(2):
            virtual = source.copy()
            virtual[axis] = 2*side*dimensions[axis]-source[axis]
            images.append((virtual,reflections[axis*2+side]))
    paths = [(np.linalg.norm(point-mic),pressure) for point,pressure in images]
    arrivals = [(distance*16000/343,pressure/distance) for distance,pressure in paths]
    impulse = np.zeros(int(max(t for t,_ in arrivals))+2)
    for time,amp in arrivals:
        offset = int(time)
        fraction = time-offset
        impulse[offset:offset+2] += [amp*(1-fraction),amp*fraction]
    return impulse,arrivals[0][0]


def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve
    import check_bella_plus2_playback as study
    import willow_room_reference as room

    p,daily = study.P,study.daily
    prior = p.read(room.PLAN)
    files = {s:prior["files"][s] for s in prior["runtime"].values()}
    conversation = ROOT/"results/4090-chatterbox-daily-training-admission-v1.json"
    spec = ROOT/"experiments/bella_prospective_room_protocol.md"
    files.update({str(f):p.sha(f) for f in (SCRIPT,spec,conversation,daily.matched.PACK,daily.board.weak.PACK,
                 study.CUE,daily.board.bella.CUE,daily.matched.PLAN,study.OUT,Path(room.official.__file__),room.official.SOURCE)})
    paired = p.read(daily.matched.PLAN)
    assert p.sha(daily.matched.PACK)==paired["files"][str(daily.matched.PACK)] and p.sha(daily.board.weak.PACK)==daily.board.run_guest.BELLA_SHA
    original,sr=sf.read(daily.board.bella.CUE,dtype="int16")
    plus,cr=sf.read(study.CUE,dtype="int16")
    assert sr==cr==16000 and np.array_equal(plus,np.rint(original.astype(float)*10**.1))
    distractors = [r for r in p.read(conversation)["admitted"] if r["request"]["group"]=="conversation"]
    rng = np.random.Generator(np.random.PCG64(SEED))
    WORK.mkdir(parents=True,exist_ok=True)
    rows,rooms=[],[]
    with ZipFile(daily.matched.PACK) as willows, ZipFile(daily.board.weak.PACK) as nyxes:
        witems=json.loads(willows.read("manifest.json"))["records"]
        nyx_items={r["file"]:r for r in json.loads(nyxes.read("manifest.json"))["records"]}
        for i,entry in enumerate(witems[:12]):
            n=nyx_items[entry["file"]]
            w=daily.study.playback.read_wav(willows,entry)
            y=daily.study.playback.read_wav(nyxes,n)
            end=entry["command_at"]
            assert n["command_at"]==end and np.array_equal(w[end:],y[end:])
            assert np.array_equal(y[end-len(plus):end],original)
            dims=rng.uniform([3.5,4,2.4],[7,8,3.4])
            def position():
                return rng.uniform([.5,.5,.9],[dims[0]-.5,dims[1]-.5,1.65])
            mic,speaker,other=position(),position(),position()
            wall=rng.uniform(.35,.75,6)
            hx,direct=image_response(dims,mic,speaker,wall)
            hi,other_direct=image_response(dims,mic,other,wall)
            cue_nyx=y.astype(float); cue_nyx[end:]=0
            cue_willow=w.astype(float); cue_willow[end:]=0
            command=y.astype(float); command[:end]=0
            length=len(y)+max(len(hx),len(hi))+8000
            def wet(x,h):
                sig=fftconvolve(x,h)
                return np.pad(sig,(0,max(0,length-len(sig))))[:length]
            def active_rms(x):
                active=x[np.abs(x) >= .1*np.max(np.abs(x))]
                return float(np.sqrt(np.mean(active**2)))
            noise=rng.normal(0,12,length)
            speech=distractors[i]
            assert speech["request"]["role"]=="training-only"
            path=Path(speech["wav"]["path"])
            assert p.sha(path)==speech["wav"]["sha256"]
            files[str(path)]=p.sha(path)
            talk,tsr=sf.read(path,dtype="int16")
            assert tsr==16000
            active=np.flatnonzero(np.abs(talk.astype(float))>=.1*np.max(np.abs(talk.astype(float))))
            center=(n["cue_file_start"]+end)//2
            start=center-int((active[0]+active[-1])//2)+round(direct-other_direct)
            assert start>=0
            interfering=wet(np.pad(talk,(start,0)),hi)
            db=(-6,0,6)[i%3]
            interferer_gain=active_rms(wet(cue_nyx,hx))/active_rms(interfering)*10**(-db/20)
            interfering*=interferer_gain
            rooms.append(dict(room=i,dimensions=dims.tolist(),mic=mic.tolist(),speaker=speaker.tolist(),
                         competitor=other.tolist(),wall=wall.tolist(),target_direct_samples=direct,
                         competitor_direct_samples=other_direct,rir_sha256=p.digest(hx),other_rir_sha256=p.digest(hi),
                         noise_sha256=p.digest(noise),competitor_gain=interferer_gain,competitor_start=start,
                         text=speech["request"]["text"],source=str(path),active_rms_ratio_db=db,command_file=entry["file"],
                         dry_suffix_sha256=p.digest(y[end:])))
            for mode,background in (("quiet",noise),("talk",noise+.35*interfering)):
                for name,dry in (("nyx",cue_nyx),("willow",cue_willow),("erased",np.zeros_like(cue_nyx))):
                    for level in ((0,) if name=="erased" else (0,2)):
                        scaled=np.rint(dry*10**(level/20))
                        assert np.max(np.abs(scaled))<32768
                        if (name,level)==("nyx",2): assert np.array_equal(scaled[end-len(plus):end],plus)
                        # Fixed 0.35 microphone gain across all rooms; microphone noise is in ADC units.
                        analog=.35*wet(command+scaled,hx)+background
                        adc=np.rint(analog)
                        clips=int(np.count_nonzero((adc>32767)|(adc< -32768)))
                        pcm=adc.clip(-32768,32767).astype("<i2")
                        label=f"{i:02}/{mode}/{name}-{level}"
                        target=WORK/(label.replace("/","-")+".wav")
                        sf.write(target,pcm,16000,subtype="PCM_16")
                        files[str(target)]=p.sha(target)
                        start_clock=int(np.flatnonzero(scaled)[0])+int(direct) if name!="erased" else None
                        rows.append(dict(room=i,label=label,mode=mode,role=name,level=level,start=start_clock,
                              end=end+int(direct),command=n["command"],samples=len(pcm),path=str(target),
                              sha256=files[str(target)],pcm_sha256=p.digest(pcm),clipped=clips))
    assert len(rows)==len({r["pcm_sha256"] for r in rows})==120
    doc=dict(records=rows,rooms=rooms,files=files,runtime=prior["runtime"],threshold_requests=["DEFAULT"],
             seed=SEED,scope="Synthetic new direct path and six wall reflections. Noise RMS 12 ADC counts. "
             "Fixed microphone gain 0.35 for target and competitor, 10%-peak active RMS on isolated wet signals.")
    if PLAN.exists():
        assert p.read(PLAN)==doc
    else:
        p.save(PLAN,doc)


def board():
    import soundfile as sf
    import check_address_gain_wakes as comparison
    import willow_room_reference as room
    p4=comparison.fixed
    plan=room.read(PLAN)
    previous=room.read(p4.REPORT)
    image=comparison.fixed_image(previous,"fixed")
    cache=WORK/"p4"/comparison.digest(bytes.fromhex(previous["qemu_sha256"])+image[:2<<20])
    cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    with TemporaryDirectory(dir=WORK) as tmp:
        stage=Path(tmp)
        with ZipFile(stage/"p4.zip","w",ZIP_DEFLATED) as uart:
            for r in plan["records"]:
                mic,sr=sf.read(r["path"],dtype="int16")
                assert sr==16000 and comparison.digest(mic.tobytes())==r["pcm_sha256"]
                block=p4.run_guest.payload(mic.tobytes(),bytes(mic.nbytes))
                for attempt in range(2):
                    try:
                        obs,log,failed=p4.boot(cache,image,cache,block,1)
                        break
                    except AssertionError:
                        if attempt: raise
                faults={}
                for path in failed:
                    name="failed/"+r["label"]+"/"+path.name
                    uart.write(path,name)
                    faults[name]=comparison.digest(path.read_bytes())
                uart.write(log,r["label"]+".log")
                rows.append(dict(label=r["label"],events=obs[0]["events"],uart_sha256=comparison.digest(log.read_bytes()),
                                 input_sha256=comparison.digest(block),failed=faults))
                if len(rows)%20==0: print(len(rows),flush=True)
        room.save(stage/"p4.json",dict(records=rows,guest_sha256=previous["guest_sha256"]["fixed"],
                  qemu_sha256=previous["qemu_sha256"],archive_sha256=comparison.digest((stage/"p4.zip").read_bytes())))
        for name in ("p4.zip","p4.json"):
            (stage/name).replace(WORK/name)


def report():
    import willow_room_reference as room
    p4=room.read(WORK/"p4.json");ref=room.read(WORK/"willow.json")
    plan=room.read(PLAN)
    groups=defaultdict(lambda:defaultdict(Counter))
    rows=[]
    for r,n,w in zip(plan["records"],p4["records"],ref["records"],strict=True):
        assert r["label"]==n["label"]==w["label"] and r["pcm_sha256"]==w["pcm_sha256"]
        for model,times in (("nyx",n["events"]["O"]),("willow",[e[1] for e in w["events"]])):
            first=next(iter(times),None)
            for cohort in (r["mode"],r["mode"]+"/"+str(plan["rooms"][r["room"]]["active_rms_ratio_db"])):
                c=groups[cohort][model+"/"+r["role"]+"/"+str(r["level"])]
                timely=first is not None and r["start"] is not None and r["start"]<=first<r["end"]
                early=first is not None and r["start"] is not None and first<r["start"]
                c.update(cases=1,opened=int(first is not None),timely=int(timely),early=int(early))
        rows.append(dict(**r,nyx=n,willow=w))
    with TemporaryDirectory(dir=WORK) as tmp:
        stage=Path(tmp); archive=OUT.with_suffix(".zip")
        with ZipFile(stage/archive.name,"w",ZIP_DEFLATED) as out:
            for name,meta in (("nyx",p4),("willow",ref)):
                src=WORK/(("p4" if name=="nyx" else "willow")+".zip")
                assert room.digest(src.read_bytes())==meta["archive_sha256"]
                with ZipFile(src) as z:
                    for f in z.namelist(): out.writestr(name+"/"+f,z.read(f))
        room.save(stage/OUT.name,dict(records=rows,counts=groups,plan_sha256=room.digest(PLAN.read_bytes()),guest_sha256=p4["guest_sha256"],
             qemu_sha256=p4["qemu_sha256"],archive_sha256=room.digest((stage/archive.name).read_bytes())))
        (stage/archive.name).replace(archive)
        (stage/OUT.name).replace(OUT)


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="case":
        sys.path.insert(0,str(ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN,official.__file__=PLAN,str(SCRIPT)
        official.case()
    elif action=="willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    else:
        {"prepare":prepare,"p4":board,"report":report}[action]()
