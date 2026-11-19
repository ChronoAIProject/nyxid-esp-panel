"""Pair panel levels across other development tracks and room paths with an independent source."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_marker_gain_music as music
import complete_quieter_panel_matrix as panel

ch, tx = panel.ch, panel.tx
OUT = ch.RESULTS / "4090-c161-panel-music-transfer.json"


def room_mixes(external_cue=None):
    songs = ch.read(music.fit.DEV)["records"][1:]
    rooms = sorted(music.ROOMS)
    v5, original = tx.study.panel_wave()
    quiet, rate = sf.read(panel.demo.WAVE, dtype="int16")
    assert rate == 16000 and len(rooms) >= 22 and len(songs)==11
    bank = tx.study.grid.room.rooms.source_bank()
    sources = sorted(k for k in bank if k[1].startswith(("Raise ", "Read ")))
    assert len(sources) == 8
    local_at = 32000
    for j, song in enumerate(songs):
        hp, local_room, local_sha = tx.study.grid.room.rooms.read_room(f"h{rooms[2*j]:03}_")
        hx, external_room, external_sha = tx.study.grid.room.rooms.read_room(f"h{rooms[2*j+1]:03}_")
        path = Path(song["path"])
        assert ch.sha(path) == song["sha256"]
        voice, text = sources[j % len(sources)]
        command, _ = tx.balanced.match_level(v5, bank[voice,text])
        wet = fftconvolve(np.r_[v5,command].astype(float), hx)
        wet_cmd = fftconvolve(np.r_[np.zeros_like(v5),command].astype(float), hx)
        if external_cue is not None:
            assert external_cue.shape == v5.shape
            alternate = fftconvolve(np.r_[external_cue,command].astype(float), hx)
        ref_rms = np.sqrt(np.mean(fftconvolve(v5.astype(float), hp)[:len(v5)]**2))
        panels = {name:fftconvolve(w.astype(float),hp) for name,w in (("original",original),("quiet",quiet))}
        longest = local_at + max(tx.study.OFFSETS) + len(wet) + 4800
        with sf.SoundFile(path) as f:
            assert f.samplerate == 16000 and f.channels == 1
            offset = (2654435761*(j+1)) % (len(f)-longest+1)
            f.seek(offset)
            bg = f.read(longest)
        bg *= 0.35*ref_rms/np.sqrt(np.mean(bg**2))
        for k, lag in enumerate(tx.study.OFFSETS):
            start = local_at+lag
            length = start+len(wet)+4800
            for db in (-6,0,6):
                views = {}
                for name, local in panels.items():
                    combined = bg[:length].copy()
                    combined[local_at:local_at+len(local)] += local
                    for role, ext in (("both",wet),("no_address",wet_cmd)):
                        signal = combined.copy()
                        signal[start:start+len(ext)] += ext*10**(db/20)
                        views[name,role] = signal
                gain = min(1.,28000/max(np.abs(v).max() for v in views.values()))
                if external_cue is not None:
                    # Keep the exact preamp gain of the original scene. Rebuild the
                    # alternate positive from its sources to avoid requantizing PCM.
                    other = bg[:length].copy()
                    other[local_at:local_at+len(panels["quiet"])] += panels["quiet"]
                    other[start:start+len(alternate)] += alternate*10**(db/20)
                    views["quiet_rms","both"] = other
                meta = dict(track=j, music=song["source_path"], noise_sha256=song["sha256"], offset=offset,
                    panel_room=local_room, external_room=external_room, panel_sha256=local_sha,
                    ext_room_sha256=external_sha, voice=voice, command=text, command_pcm_sha256=ch.digest(command),
                    gain=float(gain), external_db=db, panel_at=local_at, start=start,
                    command_at=start+len(v5), lag=lag, assigned=(k==j%len(tx.study.OFFSETS)))
                yield meta, views


def inputs(external_cue=None):
    for meta, views in room_mixes(external_cue):
        yield meta, {key:tx.study.grid.room.quantize(v*meta["gain"]) for key,v in views.items()}


def main():
    current = ch.read(panel.OUT)
    assert tx.panel.grid.gap.uplink.firmware() == current["codec"]
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == current["firmware"]
    small, identity = ch.load_reader()
    assert identity == current["small"]
    turbo = tx.diag.reader(current["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    marker = tx.event_time()
    rows, guest, summary = [], [], defaultdict(Counter)
    scratch = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(scratch, "w", ZIP_DEFLATED) as z:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for i, (s, views) in enumerate(inputs()):
            clock = s["panel_at"]+marker
            for (name,role), x in views.items():
                raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, x)
                zero = np.zeros_like(x)
                expected = panel.middle.pair.admit(raw, clock, panel.middle.single.hits(x, clock), zero)
                times = expected["opened"]
                timely = bool(times and s["start"]<=times[0]<s["command_at"])
                rows.append(dict(scene=i, source=s, panel=name, role=role, result=expected, timely=timely))
                key = f'{name}.{role}.{s["external_db"]}'
                summary["host."+key].update(trials=1, opens=int(bool(times)), timely=int(timely))
                if s["assigned"] and (role=="both" or (name=="quiet" and s["external_db"]==-6)):
                    case = (i, s, "bca", "both", x, zero, expected)
                    row = tx.panel.capture(work, z, len(guest), case, small, turbo, decoder, builds, tx_pair=clock)
                    guest.append(dict(row, panel=name, view=role, timely=timely, transmit_event=clock))
                    summary["guest."+key].update(trials=1, opens=int(bool(times)), timely=int(timely),
                                                  small=int(row["small"]), turbo=int(row["turbo"]))
            if (i+1)%15 == 0:
                print("tracks", (i+1)//15, "guest", len(guest), flush=True)
    assert (len(rows),len(guest)) == (660,77)
    with ZipFile(scratch) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    archive = OUT.with_suffix(".zip")
    scratch.replace(archive)
    ch.save(OUT, dict(records=rows, guest=guest, summary=summary, archive_sha256=ch.sha(archive),
        model=current["model"], firmware=current["firmware"], codec=current["codec"], small=identity, turbo=current["turbo"],
        sources={p.name:ch.sha(p) for p in (panel.OUT,panel.demo.WAVE,music.fit.DEV,Path(__file__))},
        policy="Pair the original and quieter local demo on 11 previously consumed music tracks, each with "
               "two separate development room responses, one of eight existing voice-command pairs, five offsets, "
               "three external gains and an external no-address control. Use the same preamp gain for all views. "
               "P4 covers one offset per track fixed by track index at all three gains on each panel version plus "
               "the matching quiet-panel weakest-gain no-address control. No new independent FAR exposure."))


if __name__ == "__main__":
    main()
