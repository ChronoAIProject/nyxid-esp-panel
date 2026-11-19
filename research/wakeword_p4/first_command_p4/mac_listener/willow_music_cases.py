"""Prepare and run official Willow on the eleven assigned room and music command clocks."""
import sys
from pathlib import Path

ROOT=Path("research/wakeword_p4")
PLAN=ROOT/"results/4090-c161-willow-marker-music-plan.json"
WORK=Path(".local/wakeword/c161-willow-marker-music")
RUNNER=Path(__file__).resolve()


def prepare():
    import io
    from fractions import Fraction
    from zipfile import ZipFile

    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve

    import check_marker_gain_music as study
    from train_c161_music_proposals import RESULTS

    ch,grid=study.ch,study.grid
    old=ch.read(grid.room.rooms.PLAN)
    history=ch.read(study.OUT)
    music=[r for r in history["records"] if r["group"]=="other_music"]
    songs={r["source_path"]:r for r in ch.read(study.fit.DEV)["records"]}
    dry=study.tracks()
    bank=grid.room.rooms.source_bank()
    dry_zip=grid.room.rooms.DRY
    assert ch.sha(dry_zip)==old["files"][str(dry_zip)]
    known_runtime=old["runtime"]
    files={k:old["files"][k] for k in known_runtime.values()}
    import run_c161_willow_matched_scenes_v1 as launcher
    for p in (RUNNER,Path(launcher.__file__),Path(launcher.official.__file__),launcher.official.SOURCE,dry_zip,study.OUT):
        files[str(p)]=ch.sha(p)
    rows=[]
    WORK.mkdir(exist_ok=True)
    with ZipFile(dry_zip) as z:
        for i,r in enumerate(music):
            ri, rem=divmod(i,22)
            song,level=divmod(rem,2)
            if ri!=(3*song+1)%28 or level!=song%2: continue
            s=r["source"]
            track=songs[s["track"]]
            assert s["noise_sha256"]==track["sha256"]==ch.sha(Path(track["path"]))
            with sf.SoundFile(track["path"]) as f:
                f.seek(s["offset"])
                common=f.read(s["samples"])*s["noise_gain"]
            h,_,checksum=grid.room.rooms.read_room(f'h{s["room_id"]:03}_')
            assert checksum==s["rir_sha256"]
            cmd=bank[s["voice"],s["command"]]
            assert ch.digest(cmd)==s["command_pcm_sha256"]
            wet=fftconvolve(cmd.astype(float),h)
            end=s["command_at"]
            common[end:end+len(wet)]+=wet
            control=grid.room.quantize(common*s["mic_gain"])
            assert ch.digest(control)==r["control_sha256"]
            willow,rate=sf.read(io.BytesIO(z.read(s["voice"]+"-Willow.wav")),dtype="int16")
            assert rate==16000
            rms=lambda x:np.linalg.norm(x.astype(float))/np.sqrt(len(x))
            match=rms(dry["v5"])/rms(willow)
            ratio=Fraction(1000000,1000000+s["player_ppm"])
            for role,wave,gain in (("v5_identity",dry["v5"],1.),("native",willow,1.),
                                   ("matched_rms",willow,match),("no_address",np.zeros_like(willow),1.)):
                warped=grid.drift.project(wave,ratio)
                at=end-len(warped)
                response=fftconvolve(warped.astype(float),h)
                mix=common.copy()
                mix[at:at+len(response)]+=response*(10**(s["level_db"]/20)*gain)
                assert np.abs(mix*s["mic_gain"]).max()<=32767
                mic=grid.room.quantize(mix*s["mic_gain"])
                if role=="v5_identity":
                    assert ch.digest(mic)==r["routes"]["v5"]["pcm_sha256"]
                    continue
                path=WORK/f"{len(rows):03}.wav"
                sf.write(path,mic,16000,subtype="PCM_16")
                files[str(path)]=ch.sha(path)
                rows.append(dict(grid_index=i,voice=s["voice"],carrier=s["command"],room=s["room"],track=s["track"],
                    level_db=s["level_db"],cue_gain=float(gain),player_ppm=s["player_ppm"],start=at,end=end,
                    target_id=role,samples=len(mic),pcm_sha256=ch.digest(mic),sha256=files[str(path)],path=str(path),
                    original_dry_rms_ratio=float(rms(willow)/rms(dry["v5"])),control_sha256=r["control_sha256"]))
    assert len(rows)==33 and len(set(r["track"] for r in rows))==11
    plan=dict(runtime=known_runtime,records=rows,files=files,threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert ch.read(PLAN)==plan
    else:
        ch.save(PLAN,plan)


def run():
    import run_c161_willow_matched_scenes_v1 as official
    official.PLAN=PLAN
    official.OUT=WORK/"runs"
    official.__file__=str(RUNNER)
    official.willow()


def case():
    sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
    import run_willow_mac_matched_prime_v1 as official
    official.PLAN=PLAN
    official.__file__=str(RUNNER)
    official.case()


if __name__=="__main__":
    {"prepare":prepare,"run":run,"case":case}[sys.argv.pop(1)]()
