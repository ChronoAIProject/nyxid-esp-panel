"""Replay the issued weak-target 079 programs through PCM48 and AAC-LC."""
from pathlib import Path
import sys

import bella_surrey_orientations as study

ROOT = study.ROOT
WORK = Path(".local/wakeword/eight-local-player-079")
OUT = ROOT/"results/4090-eight-local-player-079.json"
PLAN = OUT.with_stem(OUT.stem+"-plan")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = ROOT/"experiments/eight_local_player_079_protocol.md"


def prepare():
    import av
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import json
    from zipfile import ZipFile

    import bella_eight_voice_codec as codec
    import early_groups_local_gain as local
    import run_willow_mac_matched_prime_v1 as official

    source, p = local.first, local.P
    level, mac = source.shifts.level, source.mac
    weak = p.read(level.OUT)
    old = {r["role"]:r for r in weak["records"] if
           (r["scene"],r["angle"],r["direction"],r["mode"]) == (79,210,"front","target-minus6")}
    report = p.read(local.OUT)
    paired = {(r["role"],r["voice"]):r for r in report["records"]}
    runtime = p.read(level.PLAN)
    files = {path:runtime["files"][path] for path in runtime["runtime"].values()}
    mat = level.angles.WORK/"angle_210.0.mat"
    deps = (SCRIPT, PROTOCOL, local.OUT, local.PACK, source.voices.PACK, mac.OUT, mac.PACK, level.OUT,
            level.PLAN, level.study.PLAN, mat, official.SOURCE, Path(codec.__file__).resolve().relative_to(Path.cwd()))
    files.update({str(path):p.sha(path) for path in deps})
    assert files[str(local.PACK)]==report["sources"][str(local.PACK)]
    assert files[str(source.voices.PACK)]==report["sources"][str(source.voices.PACK)]
    receipt = p.read(mac.OUT)
    assert files[str(mac.PACK)]==receipt["archive_sha256"] and files[str(mat)]==runtime["files"][str(mat)]
    assert files[str(level.PLAN)]==weak["plan_sha256"]
    scale = p.read(level.study.PLAN)["paths"][0]["common_scale"]
    ir = loadmat(mat)["brir"]
    paths = [resample_poly(ir[:,1,ch],1,3)/scale for ch in (2,8)]
    rng = np.random.Generator(np.random.PCG64(631820))
    sizes = {r["scene"]:r["samples"] for r in weak["records"]}
    for angle in level.angles.GIT_BLOBS:
        for side in ("front","back"):
            for scene in (77,79,90):
                draw = rng.normal(0,12,sizes[scene])
                if (angle,side,scene)==(210,"front",79):
                    noise=draw
    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    impulse = np.zeros((24000,2))
    impulse[8000] = [.75,-.5]
    for _, returned, _ in codec.player_paths(impulse):
        assert np.all(abs(returned).argmax(axis=0)==8000)
    def render(stereo):
        return codec.air.receive(32768*stereo,*paths,noise)
    sources = []
    with ZipFile(mac.PACK) as issued:
        def unpack(role):
            name = f"079/{role}.wav"
            return source.shifts.member_wave(issued,name,receipt["files"][name],"float64")
        blank = unpack("erased")
        end = json.loads(issued.read("manifest.json"))["spoken_command_at"]
        for role,bundle in (("old",source.voices.PACK),("new",local.PACK)):
            with ZipFile(bundle) as z:
                for voice,entry in json.loads(z.read("manifest.json"))["voices"].items():
                    cue = source.shifts.member_wave(z,entry["file"],entry["sha256"],"int16")
                    stereo = blank.copy()
                    stereo[end-len(cue):end,0] = cue.astype(float)*source.shifts.GAIN/131072
                    if voice=="af_bella":
                        np.testing.assert_array_equal(stereo,unpack(role))
                    baseline=paired[role,voice]
                    sources.append((role,voice,stereo,baseline["pcm_sha256"],baseline["events"],baseline["start"]))
        for role,key in (("willow","willow"),("erased","absent")):
            base=old[key]
            sources.append((role,role,unpack(role),base["pcm_sha256"],base["nyx"],base["start"]))
    rows=[]
    WORK.mkdir(exist_ok=True,parents=True)
    ref=old["willow"]["reference_path"]
    files[ref]=runtime["files"][ref]
    for role,voice,stereo,original,events,start in sources:
        direct,clip=render(stereo)
        assert not clip.any() and p.digest(direct)==original and np.max(abs(stereo))<1
        for mode,returned,meta in codec.player_paths(stereo):
            label=f"{mode}-{role}-{voice}"
            mic,clipped=render(returned)
            path=WORK/(label+".wav")
            sf.write(path,mic,16000,subtype="PCM_16")
            files[str(path)]=p.sha(path)
            rows.append(dict(label=label,role=role,voice=voice,mode=mode,start=start,end=old["willow"]["end"],
                             samples=len(mic),command=old["willow"]["command"],path=str(path),sha256=files[str(path)],
                             pcm_sha256=p.digest(mic),reference_path=ref,ref_sha256=old["willow"]["ref_sha256"],
                             parent_pcm_sha256=original,parent_events=events,clipped=int(clipped.sum()),conversion=meta))
    assert len(sources)==18 and len(rows)==len({r["label"] for r in rows})==36
    p.save(PLAN,dict(records=rows,runtime=runtime["runtime"],threshold_requests=["DEFAULT"],files=files,
                     gpu=gpu,torch=str(torch.__version__),pyav=av.__version__,ffmpeg=av.library_versions,
                     impulse_returned_sample=[8000,8000]))


if __name__=="__main__":
    stage=sys.argv.pop(1)
    if stage=="prepare":
        prepare()
    elif stage=="case":
        sys.path.insert(0,str(ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN,official.__file__=PLAN,str(SCRIPT)
        official.case()
    elif stage=="willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    else:
        study.WORK,study.PLAN,study.OUT=WORK,PLAN,OUT
        {"nyx":study.nyx,"report":study.report}[stage]()
        if stage=="report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
