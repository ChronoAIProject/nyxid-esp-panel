"""Run the frozen headroom cue and three paired controls on additional Surrey orientations."""
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import bella_surrey_orientations as study

WORK = Path(".local/wakeword/bella-headroom-new-angles")
OUT = study.OUT.with_name("4090-bella-headroom-new-angles.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTO = study.ROOT/"experiments/bella_headroom_new_angles_protocol.md"
ANGLES = {120:"92468e817421b5c34c55689fe8701f819f0a82053d6fb567da9cf76a2ee74bc0",
          240:"97528ee6700600f044a5241304dcb6be81063d8ea324634093d3b1295c6cd899"}


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf
    import bella_marker_headroom as fit
    import surrey_fixed_cues as surrey
    import check_address_gain_wakes as checks
    P=fit.P
    prev=P.read(study.PLAN)
    receipt=P.read(fit.speakers.OUT)
    files={p:prev["files"][p] for p in prev["runtime"].values()}
    uses=(SCRIPT,PROTO,fit.OUT,fit.WAV,fit.early.WAV,Path(study.__file__),study.PLAN,fit.speakers.OUT,
          checks.room.official.SOURCE,Path(checks.room.official.__file__))
    files.update({str(p):P.sha(p) for p in uses})
    assert P.sha(fit.WAV)==P.read(fit.OUT)["fit"]["cue_sha256"]=="a62dfff9d2e1beaf9dbb63f54ac88f33665082884ecf67f64150606a69f1a9c2"
    assert P.sha(fit.speakers.PACK)==receipt["archive_sha256"]
    candidate,sr=sf.read(fit.WAV,dtype="int16")
    released,rr=sf.read(fit.early.WAV,dtype="int16")
    assert sr==rr==16000 and candidate.shape==released.shape
    cases={r["scene"]:r for r in prev["records"] if r["role"]=="early-nyx"}
    src={r["index"]: r for r in P.read(fit.trial.daily.OUT)["records"] if r["role"]=="original"}
    scale=prev["paths"][0]["common_scale"]
    rng=np.random.Generator(np.random.PCG64(927463))
    records,paths=[],[]
    WORK.mkdir(exist_ok=True,parents=True)
    with ZipFile(fit.speakers.PACK) as pack:
        for angle,checksum in ANGLES.items():
            mat=WORK/f"angle_{angle}.0.mat"
            url=surrey.DATA_URL.replace("angle_0.0.mat",mat.name)
            if not mat.exists():
                with urlopen(url,timeout=60) as stream:
                    raw=stream.read()
                assert study.sha256(raw).hexdigest()==checksum
                mat.write_bytes(raw)
            assert P.sha(mat)==checksum
            files[str(mat)]=checksum
            measured=loadmat(mat)["brir"]
            assert measured.shape==(65536,2,24)
            hs={k:resample_poly(measured[:,1,c],1,3)/scale for k,c in (("front",2),("back",8))}
            for direction,h in hs.items():
                other=hs["front" if direction=="back" else "back"]
                arrival=int(np.argmax(np.abs(h[:800])))
                paths.append(dict(angle=angle,direction=direction,arrival=arrival,source_url=url,rir_sha256=P.digest(h)))
                for scene in (77,79,90):
                    end=src[scene]["command_at"]
                    talk=fit.speakers.unpack(pack.read(f"{scene:03}/interruption.wav"))
                    interference=.25*fftconvolve(talk.astype(float),other)
                    bg=interference+rng.normal(0,12,len(interference))
                    ref=np.zeros(len(bg),dtype="<i2")
                    ref_path=WORK/f"ref-{len(ref)}.wav"
                    sf.write(ref_path,ref,16000,subtype="PCM_16")
                    files[str(ref_path)]=P.sha(ref_path)
                    for role in ("early-nyx","headroom","willow","absent"):
                        dry=fit.speakers.unpack(pack.read(f"{scene:03}/{'early-nyx' if role=='headroom' else role}.wav"))
                        if role=="headroom":
                            assert np.array_equal(dry[end-len(released):end],released)
                            dry[end-len(released):end]=candidate
                        start=int(np.flatnonzero(dry[:end])[0])+arrival if role!="absent" else None
                        total=np.rint(.25*fftconvolve(dry.astype(float),h)+bg)
                        mic=total.clip(-32768,32767).astype("<i2")
                        label=f"{angle}-{direction}-{scene:03}-{role}"
                        path=WORK/(label+".wav")
                        sf.write(path,mic,16000,subtype="PCM_16")
                        files[str(path)]=P.sha(path)
                        records.append(dict(label=label,angle=angle,direction=direction,scene=scene,role=role,mode="talk",
                            command=cases[scene]["command"],start=start,end=end+arrival,samples=len(mic),path=str(path),
                            sha256=files[str(path)],pcm_sha256=P.digest(mic),reference_path=str(ref_path),
                            ref_sha256=P.digest(ref),clipped=int(np.count_nonzero(total!=mic))))
    assert len(records)==len({r["pcm_sha256"] for r in records})==48
    P.save(PLAN,dict(records=records,paths=paths,files=files,runtime=prev["runtime"],threshold_requests=["DEFAULT"]))


if __name__=="__main__":
    stage=sys.argv.pop(1)
    if stage=="prepare":
        prepare()
    elif stage=="case":
        sys.path.insert(0,str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN,official.__file__=PLAN,str(SCRIPT)
        official.case()
    elif stage=="willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    else:
        study.WORK,study.PLAN,study.OUT=WORK,PLAN,OUT
        {"nyx":study.nyx,"report":study.report}[stage]()
