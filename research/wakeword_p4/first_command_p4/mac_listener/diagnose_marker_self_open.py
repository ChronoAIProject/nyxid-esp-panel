"""Reproduce the self opening and replace cue/command independently in the microphone mix."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

import check_marker_gain_music as study

ch, prior, grid = study.ch, study.prior, study.grid
OUT = ch.RESULTS/(study.fit.STEM+"-self.json")


def main():
    records = ch.read(study.OUT)
    dry = study.tracks()
    bank = grid.room.rooms.source_bank()
    songs = {r["sha256"]:Path(r["path"]) for r in ch.read(study.fit.DEV)["records"]}
    collisions = [r for r in records["records"] if r["group"]=="other_music" and r["routes"]["markers"]["self_opened"]]
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    rows = []
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        full = ch.binding(work/"full.so",coded="v5")
        pair = ch.binding(work/"pair.so",coded="aligned")
        for r in collisions:
            s = r["source"]
            assert s["player_ppm"]==0
            h,_,rh = grid.room.rooms.read_room(f'h{s["room_id"]:03}_')
            assert rh == s["rir_sha256"]
            with sf.SoundFile(songs[s["noise_sha256"]]) as f:
                f.seek(s["offset"])
                bg = f.read(s["samples"])*s["noise_gain"]
            cmd = fftconvolve(bank[s["voice"],s["command"]].astype(float),h)
            for name,wave in dry.items():
                cue = fftconvolve(wave.astype(float),h)*10**(s["level_db"]/20)
                ref = prior.level.guard.refs(s,name,dry)
                for has_cue,has_cmd in ((True,True),(True,False),(False,True),(False,False)):
                    x = bg.copy()
                    if has_cmd:
                        x[s["command_at"]:s["command_at"]+len(cmd)] += cmd
                    if has_cue:
                        x[s["start"]:s["start"]+len(cue)] += cue
                    mic = grid.room.quantize(x*s["mic_gain"])
                    idx = len(rows)
                    for attempt in range(2):
                        try:
                            board = prior.p4.phase.probe(work,s,mic,ref,full,pair,idx)
                            break
                        except (AssertionError,TimeoutError):
                            z.write(work/"serial.log",f"retry/{idx}-{attempt}.txt")
                            if attempt: raise
                    if has_cmd and has_cue:
                        assert ch.digest(mic)==r["routes"][name]["pcm_sha256"]
                        assert board["opened"]==r["routes"][name]["self_opened"]
                    if has_cmd and not has_cue:
                        assert ch.digest(mic)==r["control_sha256"]
                    z.writestr(f"events/{idx:03}.txt","\n".join(board["uart_events"])+"\n")
                    rows.append(dict(index=r["index"],cue=name,cue_on_mic=has_cue,command_on_mic=has_cmd,board=board))
    with ZipFile(tmp) as z:
        assert len(rows)==len(collisions)*12 and z.testzip() is None
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,archive_sha256=ch.sha(archive),source_sha256=ch.sha(study.OUT),
                     script_sha256=ch.sha(Path(__file__)),firmware_sha256=ch.sha(prior.p4.phase.handoff.test.BUILD/"nyx_mac_listener.bin"),
                     policy="Every new-song self opening of the marker candidate. Four microphone counterfactuals "
                     "at identical clocks/gain and with the same digital cue reference. Repeat for both controls."))


if __name__=="__main__":
    main()
