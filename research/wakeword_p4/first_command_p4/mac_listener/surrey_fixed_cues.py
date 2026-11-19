"""Play fixed same-voice Nyx and Willow through measured front and back paths in the Surrey room."""
from pathlib import Path
import sys

import willow_music_cases as runner

ROOT = Path("research/wakeword_p4")
WORK = Path(".local/wakeword/c161-surrey-listening-room")
PLAN = ROOT / "results/4090-c161-surrey-fixed-cues-plan.json"
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())
MAT = WORK / "angle_0.0.mat"
MAT_SHA = "d944089e3c872b27dc9dcbf9112978b7fa170c483c3b240a8a3d62a3293c994a"
DATA_URL = ("https://raw.githubusercontent.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs/"
            "d36af1bbd0f4083a9f9ee678ff8f499044e851a7/IoSR_ListeningRoom_BRIRs_MATLAB/angle_0.0.mat")


def prepare():
    from io import BytesIO
    import json
    from tempfile import TemporaryDirectory
    from urllib.request import urlopen
    from zipfile import ZipFile

    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf

    import check_bella_weak_playback as weak
    import eight_voice_willow as previous

    ch, tx, study = weak.ch, weak.tx, weak.study
    WORK.mkdir(exist_ok=True)
    if not MAT.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
        assert weak.bella.gains.direction.sha256(raw).hexdigest() == MAT_SHA
        MAT.write_bytes(raw)
    assert ch.sha(MAT) == MAT_SHA
    measured = loadmat(MAT)["brir"]
    assert measured.shape == (65536,2,24)
    # Table 2 uses one-based channels 3 (front centre), 9 (back centre), and 2 (front right).
    impulses = resample_poly(measured[:,0,[2,8,1]],1,3,axis=0).T
    common_norm = np.linalg.norm(impulses[0])
    front, back, panel_ir = impulses/common_norm
    parent = ch.read(previous.PLAN)
    fixed = ch.read(weak.bella.OUT)
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    assert rate == 16000 and ch.sha(weak.PACK) == "67b54cc6493d1925914d0628e1785ca693e1b41dc5ee61e18c49f48858fe2e94"
    assert ch.sha(previous.PACK) == parent["files"][str(previous.PACK)]
    v5,_ = tx.study.panel_wave()
    bed,_,music_sha = tx.study.grid.room.rooms.read_noise("music")
    ref = fftconvolve(v5.astype(float),panel_ir)[:len(v5)]
    demo = fftconvolve(panel.astype(float),panel_ir)
    files = {p:parent["files"][p] for p in parent["runtime"].values()}
    for p in (RUNNER,MAT,weak.PACK,previous.PACK,weak.bella.CUE,weak.bella.OUT,previous.PLAN,study.mixed.panel.demo.WAVE):
        files[str(p)] = ch.sha(p)
    sources = {(r["batch"],r["source"]["index"]):r for r in fixed["records"] if r["paired_back_grid"]}
    native = tx.panel.grid.room.rooms.c161_binding()
    rows = []
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(weak.PACK) as issued, ZipFile(previous.PACK) as comparison:
        matcher = tx.single.timed.binding(Path(folder))
        records = json.loads(comparison.read("manifest.json"))["records"]
        cues = {}
        for role in ("nyx_marked","nyx_plain","willow_matched","no_address"):
            anchor, = (r for r in records if r["batch"]=="second" and r["track"]==0 and r["role"]==role)
            data = comparison.read(anchor["file"])
            assert weak.bella.gains.direction.sha256(data).hexdigest()==anchor["wav_sha256"]
            track,rate = sf.read(BytesIO(data),dtype="int16")
            assert rate==16000 and ch.digest(track)==anchor["pcm_sha256"]
            cues[role] = track[32000:anchor["command_at"]], anchor["command_at"]-anchor["start"]
        released,rate = sf.read(weak.bella.CUE,dtype="int16")
        assert rate==16000 and np.array_equal(released,cues["nyx_marked"][0]) and not cues["no_address"][0].any()
        def record(group, role, mic, start, end, tx_clock, clip, **meta):
            p = WORK / f"{len(rows):03}.wav"
            sf.write(p,mic,16000,subtype="PCM_16")
            files[str(p)] = ch.sha(p)
            route = study.playback.host_result(matcher,native,mic,tx_clock)
            rows.append(dict(mode=group,target_id=role,start=start,end=end,path=str(p),sha256=files[str(p)],
                pcm_sha256=ch.digest(mic),samples=len(mic),transmit_event=tx_clock,clipped=clip,nyx=route,**meta))
        for role,(cue,duration) in cues.items():
            mic = np.pad(cue,(32000,32000))
            end = 32000+len(cue)
            record("dry",role,mic,end-duration,end,0,0)
        entries = json.loads(issued.read("manifest.json"))["records"]
        assert len(entries)==24 and len({r["command_voice"] for r in entries[::3]})==8
        for entry in entries[::3]:
            name = Path(entry["file"])
            s = sources[name.parent.name,int(name.stem)]["source"]
            data = issued.read(entry["file"])
            track,rate = sf.read(BytesIO(data),dtype="int16")
            assert rate==16000 and weak.bella.gains.direction.sha256(data).hexdigest()==entry["sha256"]
            end = s["command_at"]
            assert ch.digest(track[end:])==entry["command_pcm_sha256"] and s["music_sha256"]==music_sha
            size = max(len(track)+len(front)-1,s["panel_at"]+len(demo))+4800
            bg = bed[(s["music_offset"]+np.arange(size))%len(bed)]
            mix = bg*(.35*np.sqrt(np.mean(ref**2))/np.sqrt(np.mean(bg**2)))
            at = s["panel_at"]
            mix[at:at+len(demo)] += demo
            for role,(cue,duration) in cues.items():
                source = track.copy()
                source[:end-len(cue)] = 0
                source[end-len(cue):end] = cue
                assert role!="nyx_marked" or np.array_equal(source,track)
                for position,ir in (("front",front),("back",back)):
                    wet = fftconvolve(source.astype(float),ir)*10**(s["external_db"]/20)
                    mic,clipped = study.mixed.digitize(mix+np.pad(wet,(0,size-len(wet))),s["preamp_gain"])
                    record(position,role,mic,end-duration,end,s["transmit_event"],clipped,
                        batch=name.parent.name,track=int(name.stem),carrier=s["command"],voice=s["voice"],
                        external_db=s["external_db"],source_sha256=ch.digest(source),command_sha256=entry["command_pcm_sha256"],
                        music_offset=s["music_offset"],panel_at=at,preamp=s["preamp_gain"])
    assert len(rows)==68
    plan = dict(records=rows,runtime=parent["runtime"],threshold_requests=["DEFAULT"],files=files,nyx_model=parent["nyx_model"],
        acoustics=dict(citation="Jon Francombe, IoSR Listening Room Multichannel BRIR dataset, 2017",
            doi="https://doi.org/10.15126/surreydata.00813511",license="https://creativecommons.org/licenses/by/4.0/",
            source_url=DATA_URL,source_sha256=MAT_SHA,left_ear=0,matlab_channels={"front":3,"back":9,"panel":2},
            sample_rate=48000,target_rate=16000,front_l2=float(common_norm),music_sha256=music_sha,
            impulse_sha256=[ch.digest(h.astype("<f8")) for h in (front,back,panel_ir)]),
        policy="First track per command voice at its preassigned level. Room transfer to the left ear of a fixed "
               "head simulator. Same global front L2 normalization retains relative levels and measured delays. "
               "Fixed af_bella cues right-aligned before byte-identical command suffixes, real electrical R absent. "
               "Primary timing uses sender command start. Report all methods and controls.")
    if PLAN.exists():
        assert plan==ch.read(PLAN)
    else:
        ch.save(PLAN,plan)


if __name__=="__main__":
    runner.PLAN,runner.WORK,runner.RUNNER = PLAN,WORK,RUNNER
    {"prepare":prepare,"run":runner.run,"case":runner.case}[sys.argv.pop(1)]()
