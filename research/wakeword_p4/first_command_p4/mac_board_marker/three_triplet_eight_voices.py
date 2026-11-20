"""Fit the two early Bella marker groups into eight right-aligned synthetic addresses."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import bella_eight_voice_level as previous

transfer, study = previous.transfer, previous.study
WORK = Path(".local/wakeword/three-triplet-eight-voices")
OUT = transfer.OUT.with_name("4090-three-triplet-eight-voices.json")
PLAN = OUT.with_stem(OUT.stem+"-plan")
PACK = OUT.with_name("hye-nyx-eight-three-triplet-voices.zip")
PROTOCOL = study.PROTOCOL.with_name("three_triplet_eight_voices_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def build():
    import numpy as np
    import soundfile as sf
    import torch
    import check_third_triplet_collisions as winner
    import package_bella_6db_stereo as stereo

    p = winner.P
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    WORK.mkdir(parents=True, exist_ok=True)
    plan = p.read(transfer.PLAN)
    assert p.sha(transfer.GAINS) == plan["files"][str(transfer.GAINS)]
    selected = winner.triple.wave(winner.NAME)
    assert p.sha(selected) == p.read(winner.triple.OUT)["voices"][winner.NAME]
    final, rate = sf.read(selected, dtype="int16")
    single = winner.fit.early.WAV.with_name("hye-nyx-af-bella-plus5db.wav")
    assert p.sha(single) == plan["files"][str(single)]
    base, sr = sf.read(single, dtype="int16")
    assert sr == rate == 16000 and len(base) == len(final) == 23200
    increment = torch.tensor(final.astype(float)-base, device="cuda")
    sources = {str(f):p.sha(f) for f in (transfer.GAINS, transfer.PLAN, selected, single, PROTOCOL, SCRIPT)}
    info = {}
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(transfer.GAINS) as gains:
        path = Path(tmp)/PACK.name
        with ZipFile(path,"w") as dest:
            voices = {r["voice"]:r for r in json.loads(gains.read("manifest.json"))["records"] if "wake_only" in r}
            for voice, r in voices.items():
                blob = gains.read(r["wake_only"]["file"])
                assert sha256(blob).hexdigest() == r["wake_only"]["sha256"]
                dry, sr = sf.read(BytesIO(blob), dtype="int16")
                assert sr == 16000 and len(dry) <= len(final)
                gain = plan["voices"][voice]["scales"]["single"]
                carrier = np.zeros(len(final), dtype=np.float64)
                carrier[-len(dry):] = np.rint(dry.astype(float)*gain)
                old = torch.tensor(carrier, device="cuda")
                extra = increment * (r["prefix_scale"]/voices["af_bella"]["prefix_scale"]) * (gain/plan["voices"]["af_bella"]["scales"]["single"])
                pos, neg = extra>0, extra<0
                headroom = min(1., float(((32767-old[pos])/extra[pos]).min()), float(((-32768-old[neg])/extra[neg]).min()))
                total = torch.round(old+headroom*extra).cpu().numpy()
                assert 0 <= headroom <= 1 and total.min()>=-32768 and total.max()<=32767
                pcm = total.astype("<i2")
                assert np.array_equal(pcm[final==base], carrier[final==base])
                buf = BytesIO()
                sf.write(buf, pcm, 16000, format="WAV", subtype="PCM_16")
                raw = buf.getvalue()
                if voice == "af_bella":
                    assert np.array_equal(carrier,base) and raw == selected.read_bytes()
                name = voice+".wav"
                stereo.write_member(dest,name,raw)
                info[voice] = dict(file=name, bytes=len(raw), sha256=sha256(raw).hexdigest(), pcm_sha256=p.digest(pcm),
                                   headroom_gain=headroom, prepend_samples=len(final)-len(dry))
            assert len(info)==8
            manifest = dict(voices=info, sources=sources, sample_rate=16000, samples=len(final), gpu=torch.cuda.get_device_name())
            stereo.write_member(dest,"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
        path.replace(PACK)


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    import soundfile as sf
    import package_bella_6db_stereo as stereo

    p, controls, air = stereo.P, stereo.sources, transfer.air
    old = p.read(transfer.PLAN)
    previous_reports = [p.read(path) for path in (transfer.OUT,previous.OUT)]
    for result, path in zip(previous_reports,(transfer.OUT,previous.OUT),strict=True):
        assert p.sha(path.with_suffix(".zip")) == result["archive_sha256"]
    assert p.sha(stereo.PACK) == p.read(stereo.OUT)["archive_sha256"]
    files = {f:old["files"][f] for f in old["runtime"].values()}
    for dep in (PACK, transfer.PACK, transfer.OUT, previous.OUT, transfer.PLAN, stereo.PACK, stereo.OUT, SCRIPT, PROTOCOL):
        files[str(dep)] = p.sha(dep)
    mat = controls.angles.WORK/"angle_240.0.mat"
    assert p.sha(mat) == controls.angles.ANGLES[240]
    brir = loadmat(mat)["brir"]
    scale = p.read(controls.fit.study.PLAN)["paths"][0]["common_scale"]
    surrey = [resample_poly(brir[:,1,ch],1,3)/scale for ch in (2,8)]
    noise = next(n for r,_,_,_,n in controls.stems_for({240},("early-nyx",)) if r["direction"]=="front" and r["scene"]==77)
    with ZipFile(stereo.PACK) as z:
        blank, rate = sf.read(BytesIO(z.read("077/absent-baseline.wav")))
    assert rate==16000
    ir = [sf.read(air.response_file(e))[0][:,1] for e in air.RESPONSES["aula"]]
    hall = [v/np.linalg.norm(ir[0]) for v in ir]
    hall_noise = np.random.Generator(np.random.PCG64(209473)).normal(0,12,len(blank)+20000)
    rows = []
    with ZipFile(PACK) as z, ZipFile(transfer.PACK) as paired:
        meta = json.loads(z.read("manifest.json"))["voices"]
        placements = (("surrey", "original", 1., surrey, noise, previous_reports[0]),
                      ("surrey", "plus6", 10**.3, surrey, noise, previous_reports[1]),
                      ("aula", "original", 1., hall, hall_noise, previous_reports[0]))
        for room, mode, gain, paths, background, measured in placements:
            delay = int(np.abs(paths[0][:800]).argmax())
            for r in measured["records"]:
                if r["role"] != "double" or r["room"] != room:
                    continue
                voice = r["voice"]
                end = r["end"]-delay
                source = blank.copy()
                mic = None
                for store, manifest in ((paired,old["voices"]), (z,meta)):
                    raw = store.read(manifest[voice]["file"])
                    assert sha256(raw).hexdigest()==manifest[voice]["sha256"]
                    cue, sr = sf.read(BytesIO(raw), dtype="int16")
                    assert sr==16000 and p.digest(cue)==manifest[voice]["pcm_sha256"]
                    source[:,0] = blank[:,0]
                    source[end-len(cue):end,0] = cue.astype(float)/131072
                    source[:,0] *= gain
                    mic, clipped = air.receive(32768*source,*paths,background)
                    if store is paired:
                        assert p.digest(mic)==r["pcm_sha256"]
                label = f"{room}-{mode}-{voice}-three-triplets"
                path = WORK/(label+".wav")
                sf.write(path,mic,16000,subtype="PCM_16")
                files[str(path)] = p.sha(path)
                ref = r["reference_path"]
                files[ref] = old["files"][ref]
                rows.append(dict(label=label,room=room,mode=room+"-"+mode,voice=voice,role="three-triplets",
                                 start=int(np.flatnonzero(source[:end,0])[0])+delay,end=r["end"],samples=len(mic),
                                 command=r["command"],clipped=int(clipped.sum()),path=str(path),sha256=files[str(path)],
                                 pcm_sha256=p.digest(mic),reference_path=ref,ref_sha256=r["ref_sha256"],
                                 parent_label=r["label"],parent_pcm_sha256=r["pcm_sha256"],parent_nyx=r["nyx"]))
    assert len(rows)==len({r["label"] for r in rows})==24
    p.save(PLAN,dict(records=rows,files=files,runtime=old["runtime"],threshold_requests=["DEFAULT"]))


if __name__=="__main__":
    stage = sys.argv.pop(1)
    if stage == "case":
        sys.path.insert(0,str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    elif stage in ("build","prepare"):
        {"build":build,"prepare":prepare}[stage]()
    else:
        study.WORK,study.PLAN,study.OUT=WORK,PLAN,OUT
        {"nyx":study.nyx,"report":study.report}[stage]()
        if stage=="report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
