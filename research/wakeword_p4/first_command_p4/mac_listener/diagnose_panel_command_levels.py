"""Measure the external command at the microphone and compare clean, music and received speech."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import fftconvolve

import check_panel_marker_p4 as panel

ch, study = panel.ch, panel.study
BOARD = ch.RESULTS / "4090-c161-panel-marker-order-timed-p4.json"
OUT = ch.RESULTS / "4090-c161-panel-command-level-diagnostic.json"


@lru_cache(maxsize=1)
def sources():
    v5, bca = study.panel_wave()
    noise, _, nh = study.grid.room.rooms.read_noise("music")
    return v5, bca, noise, nh, study.grid.room.rooms.source_bank()


@lru_cache(maxsize=4)
def room(geometry):
    a,b = study.GEOMETRIES[geometry]
    return [study.grid.room.rooms.read_room(f"h{i:03}_") for i in (a,b)]


def scene(row):
    s = row["source"]
    v5, bca, noise, nh, bank = sources()
    (hp, pname, ph), (hx, xname, xh) = room(s["geometry"])
    assert (pname,ph,xname,xh,nh) == tuple(s[k] for k in
            ("panel_room","panel_rir_sha256","external_room","external_rir_sha256","noise_sha256"))
    command = bank[s["voice"], s["command"]]
    assert ch.digest(command) == s["command_pcm_sha256"]
    ext = fftconvolve(np.r_[v5, command].astype(float), hx)
    local = fftconvolve(bca.astype(float),hp)
    origin, at, end = s["local_at"], s["start"], s["command_at"]
    n = at + len(ext) + 4800
    bg = noise[(39413*s["geometry"] + np.arange(n)) % len(noise)]
    panel_base = fftconvolve(v5.astype(float), hp)[:len(v5)]
    bg = bg * (.35*np.linalg.norm(panel_base)/np.sqrt(len(v5))/np.sqrt(np.mean(bg**2)))
    base = bg.copy()
    base[origin:origin+len(local)] += local
    gain = 10**(s["external_db"]/20)
    base[at:at+len(ext)] += ext*gain
    quantize = study.grid.room.quantize
    full = quantize(base*s["gain"])
    assert ch.digest(full) == row["mic_sha256"] and end == at+len(v5)
    return full, bg, local, command, hx, gain, base


def levels(row):
    s = row["source"]
    full, bg, _, command, hx, gain, base = scene(row)
    n, end = len(full), s["command_at"]
    quantize = study.grid.room.quantize
    wet = fftconvolve(command.astype(float),hx)*gain
    # Work on the exact command-to-end interval, including the room decay and the trailing pad.
    speech = np.pad(wet,(0,n-end-len(wet)))
    interference = base[end:]-speech
    def frames(x):
        return np.pad(x,(0,-len(x)%320)).reshape(-1,320)
    c = frames(speech)
    active = np.mean(c*c,axis=1) >= .001*np.max(np.mean(c*c,axis=1))
    def db_energy(x):
        return float(10*np.log10(np.mean(frames(x)[active]**2)))
    cmd_energy = db_energy(speech)
    metric = dict(command_active_dbfs=cmd_energy+20*np.log10(s["gain"]/32768),
                  command_to_music_db=cmd_energy-db_energy(bg[end:]),
                  command_to_all_interference_db=cmd_energy-db_energy(interference),
                  active_frames=int(active.sum()))
    audio = dict(room_command=quantize(speech*s["gain"]), command_music=quantize((speech+bg[end:])*s["gain"]),
                 full_mic_from_command=full[end:], interference_only=quantize(interference*s["gain"]))
    return metric, audio


def reader(info):
    model_dir = panel.prior.prior.prior.audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir/k)==v for k,v in info["files"].items())
    return WhisperModel(str(model_dir), device="cuda", compute_type="float16",cpu_threads=1,num_workers=1)


def recognized(model, x, expected, cache):
    key = (ch.digest(x),expected)
    if key not in cache:
        reading = panel.grid.gap.uplink.recognition(model,x,expected)
        cache[key] = dict(pcm_sha256=key[0], samples=len(x), text=reading["text"], complete=reading["complete_suffix"])
    return cache[key]


def main():
    report = ch.read(BOARD)
    versions = report["versions"]
    turbo = reader(versions["turbo"])
    archive = BOARD.with_suffix(".zip")
    assert ch.sha(archive)==report["archive_sha256"]
    decoder = panel.grid.gap.uplink.decoder_api()
    rows, cache, summary = [], {}, defaultdict(Counter)
    with ZipFile(archive) as z:
        for row in report["records"]:
            if (row["name"],row["role"])!=("bca","both"):
                continue
            s = row["source"]
            metric, layers = levels(row)
            command = s["command"]
            scores = {name:recognized(turbo,x,command,cache) for name,x in layers.items()}
            if row["opened"]:
                up = row["uplink"]
                text = z.read(f'{row["index"]:02}-opus.log').decode().replace("\r\n","\n")
                for lane, name in enumerate(("received","raw_lane")):
                    packets = panel.grid.gap.uplink.packets(text,row["index"],up["samples"],lane)
                    decoded = panel.grid.gap.uplink.decode(decoder,packets)
                    scores[name] = recognized(turbo,decoded,command,cache)
                    if lane == 0:
                        assert len(packets)==up["packets"] and ch.digest(decoded)==up["pcm_sha256"]
                        assert scores[name]["text"]==up["turbo"]["text"] and scores[name]["complete"]==row["turbo"]
            totals = summary[s["external_db"]]
            totals.update(trials=1,opened=int(bool(row["opened"])))
            totals.update({name:int(v["complete"]) for name,v in scores.items()})
            rows.append(dict(index=row["index"],geometry=s["geometry"],offset=s["offset"],voice=s["voice"],
                             level=s["external_db"],command=command,opened=row["opened"],mic_sha256=row["mic_sha256"],
                             levels=metric,command_at=s["command_at"],scores=scores))
    assert len(rows)==60 and all(x["trials"]==20 for x in summary.values())
    ch.save(OUT,dict(records=rows,summary=summary,asr=versions["turbo"],sources={p.name:ch.sha(p) for p in
        (BOARD,archive,Path(study.__file__),Path(__file__))},policy="For acoustic diagnostics only, begin at the known "
        "external command boundary. Active 20 ms command frames lie within 30 dB of maximum frame power. "
        "Keep the exact room, music, tail, gain and sample length. All four command-boundary views include the "
        "three misses. Received and raw Opus lanes use all packets of the real OPENs; closed trials count as "
        "incomplete. Nominal external gain applies to the whole speaker cue plus command. No exposure added."))


if __name__ == "__main__":
    main()
