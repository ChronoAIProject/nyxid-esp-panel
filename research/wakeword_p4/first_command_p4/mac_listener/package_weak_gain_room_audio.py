"""Supply the timed masking sounds and exact digital microphone pairs for Mac playback."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import bella_weak_speaker_gain as trial
import check_bella_daily_overlap as daily

P = daily.P
NAME = "c161-bella-weak-speaker-gain-room-audio.zip"
OUT = trial.RESULTS / "4090-c161-bella-weak-speaker-gain-room-audio.json"
MUSIC = trial.RESULTS / "4090-c161-music-development-sources-v1.json"
LICENSE = Path(".local/wakeword/musan-licenses/licenses/music/fma-western-art/LICENSE")
CREDIT = ("Music from Selections from the 2005-2006 Season by Advent Chamber Orchestra, MUSAN "
          "music-fma-wa-0004. The timed excerpts and microphone mixes containing them are provided "
          "under CC BY-SA 3.0 US (https://creativecommons.org/licenses/by-sa/3.0/us/). "
          "Source attribution and album URL are in music-source-LICENSE.txt. Excerpts are time-offset, "
          "scaled and used in simulated room mixes.")


def wav(pcm,subtype):
    stream = BytesIO()
    sf.write(stream,pcm,16000,format="WAV",subtype=subtype)
    return stream.getvalue()


def main():
    source = P.read(MUSIC)
    music,_,music_hash = daily.tx.study.grid.room.rooms.read_noise("music")
    assert source["records"][0]["sha256"] == music_hash
    license_text = LICENSE.read_text()
    assert "music-fma-wa-0004" in license_text and "Selections from the 2005-2006 Season" in license_text
    grid,taps = daily.rooms()
    rooms = {case[0]["file"]:case for case in grid[-6]}
    old = {r["index"]:r for r in P.read(daily.OUT)["records"] if r["role"]=="original"}
    v5,_ = daily.tx.study.panel_wave()
    packed = []
    scenes = []
    plan = P.read(trial.PLAN)
    assert P.sha(trial.PACK)==plan["pack_sha256"]
    with TemporaryDirectory(dir=trial.WORK) as tmp, ZipFile(trial.PACK) as speakers:
        work = Path(tmp)
        output = work/NAME
        with ZipFile(output,"w",ZIP_DEFLATED) as archive:
            def add(name,data):
                daily.matched.member(archive,name,data)
                packed.append(dict(file=name,bytes=len(data),sha256=sha256(data).hexdigest()))
            add("music-source-LICENSE.txt",LICENSE.read_bytes())
            add("music-attribution.txt",(CREDIT+"\n").encode())
            entries = json.loads(speakers.read("manifest.json"))["records"]
            for i in (77,79,90):
                row = old[i]
                _,s,render,ref,_,_ = rooms[row["file"]]
                n = len(ref)
                hp,_,ph = daily.tx.diag.room(s["geometry"])[0]
                assert ph == s["panel_rir_sha256"] and s["music_sha256"] == music_hash
                section = music[(s["music_offset"]+np.arange(n)) % len(music)]
                anchor = fftconvolve(v5.astype(float),hp)[:len(v5)]
                bed = section*(.35*np.linalg.norm(anchor)/np.sqrt(len(anchor))/np.sqrt(np.mean(section**2)))
                bed *= s["preamp_gain"]
                assert 0 < np.max(np.abs(bed)) < 32767
                # Keep the music's predicted at-mic amplitude at 24-bit precision in full-scale units.
                samples = np.rint(bed*256).astype(np.int32)*256
                music_file=f"{i:03}/music-at-mic-level.wav"
                panel_file=f"{i:03}/panel-at-source-level.wav"
                add(music_file,wav(samples,"PCM_24"))
                add(panel_file,wav(ref,"PCM_16"))
                dry_zero = np.zeros(len(rooms[row["file"]][-1]["original"]))
                wet_zero = daily.board.bella.rear_filter(dry_zero,taps)
                music_panel,_ = render(wet_zero)
                panel_alone,_ = render(wet_zero,include_music=False)
                # Each component has gone through independent ADC rounding in this check.
                assert np.max(np.abs(music_panel.astype(float)-panel_alone.astype(float)-bed))<=1.000001
                scene = dict(index=i,command=row["command"],command_at=row["command_at"],panel_start=s["panel_at"],
                     scene_samples=n,reference_sha256=P.digest(ref),music_start=0,music_offset=s["music_offset"],
                     music_at_mic_rms_dbfs=float(20*np.log10(np.sqrt(np.mean(bed**2))/32768)),
                     panel_at_mic_rms_dbfs=float(20*np.log10(np.sqrt(np.mean(panel_alone.astype(float)**2))/32768)),
                     external_db=s["external_db"],panel_file=panel_file,music_file=music_file,pairs=[])
                for ent in entries:
                    if not ent["file"].startswith(f"{i:03}/"):
                        continue
                    raw=speakers.read(ent["file"])
                    assert sha256(raw).hexdigest()==ent["sha256"]
                    speaker,rate=sf.read(BytesIO(raw),dtype="int32")
                    assert rate==16000 and np.all(speaker%16384==0)
                    dry=speaker//16384
                    mic,_=render(daily.board.bella.rear_filter(dry,taps))
                    assert P.digest(mic)==ent["mic_sha256"]
                    name="digital/"+ent["file"]
                    data=wav(np.column_stack((mic,ref)),"PCM_16")
                    pair,rate=sf.read(BytesIO(data),dtype="int16")
                    assert rate==16000 and np.array_equal(pair[:,0],mic) and np.array_equal(pair[:,1],ref)
                    add(name,data)
                    scene["pairs"].append(dict(speaker=ent["file"],speaker_sha256=ent["sha256"],digital=name,
                                               mic_sha256=ent["mic_sha256"]))
                assert len(scene["pairs"])==9
                scenes.append(scene)
            manifest=dict(rate=16000,external_archive=trial.PACK.name,external_archive_sha256=P.sha(trial.PACK),
                source_music=MUSIC.name,music_source_sha256=music_hash,attribution=CREDIT,scenes=scenes,files=packed,
                playback="Start the external speaker file, music bed and timed panel file together at sample zero. "
                         "Keep each physical output level, position and ADC gain fixed across pairs. The music WAV "
                         "represents the simulated microphone level; calibrate the physical music source with a "
                         "music-only microphone recording using the full-file RMS in each scene. Play the panel "
                         "track through the panel so its own electrical reference is available. All external WAVs "
                         "use the same quarter gain. The digital stereo files contain M then R for sample-exact "
                         "injection. Record actual physical M/R and peer audio separately.")
            daily.matched.member(archive,"manifest.json",json.dumps(manifest,indent=2)+"\n")
        with ZipFile(output) as check:
            assert check.testzip() is None and len(check.namelist())==36
            for f in packed:
                assert sha256(check.read(f["file"])).hexdigest()==f["sha256"]
        P.save(work/OUT.name,dict(archive=NAME,archive_sha256=P.sha(output),bytes=output.stat().st_size,
               license=CREDIT,source_url=source["url"],scenes=scenes,sources={p.name:P.sha(p) for p in
               (Path(__file__),MUSIC,LICENSE,daily.OUT,trial.PLAN,trial.PACK)}))
        output.replace(trial.WORK/NAME)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
