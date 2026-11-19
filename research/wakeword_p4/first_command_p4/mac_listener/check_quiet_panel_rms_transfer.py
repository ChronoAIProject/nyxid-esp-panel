"""Combine the frozen equal-RMS external cue with the quieter demo across both room matrices."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_panel_music_transfer as transfer
import check_panel_rms_cue as rms

ch, tx, panel = transfer.ch, transfer.tx, transfer.panel
OUT = ch.RESULTS / "4090-c161-quiet-panel-rms-transfer.json"


def cases(cue, current):
    old = {r["host_index"]:r for r in current["records"]}
    original, _ = tx.study.panel_wave()
    demo, rate = sf.read(panel.demo.WAVE, dtype="int16")
    assert rate == 16000
    for src in ch.read(tx.diag.BOARD)["records"]:
        if (src["name"],src["role"]) != ("bca","both"):
            continue
        i, s = src["host_index"], src["source"]
        _, bg, _, cmd, hx, level, _ = tx.diag.scene(src)
        command, _ = tx.balanced.match_level(original,cmd)
        (hp,_,_),_ = tx.diag.room(s["geometry"])
        local = fftconvolve(demo.astype(float),hp)
        base = bg.copy()
        base[s["local_at"]:s["local_at"]+len(local)] += local
        for wave in (original,cue):
            mic = base.copy()
            ext = fftconvolve(np.r_[wave,command].astype(float),hx)*level
            mic[s["start"]:s["start"]+len(ext)] += ext
            x = tx.study.grid.room.quantize(mic*s["gain"])
            if wave is original:
                assert ch.digest(x)==old[i]["mic_sha256"]
        yield "first", i, s, old[i]["opened"], old[i]["transmit_event"], x, False
    records = ch.read(transfer.OUT)["records"]
    prior = {r["scene"]:r for r in records if r["role"]=="both" and r["panel"]=="quiet"}
    marker = tx.event_time()
    for i,(s,views) in enumerate(transfer.inputs(cue)):
        before = prior[i]
        assert before["source"] == s and ch.digest(views["quiet","both"]) == before["result"]["pcm_sha256"]
        yield ("music", i, s, before["result"]["opened"], s["panel_at"]+marker, views["quiet_rms","both"],
               s["assigned"] and s["external_db"]==-6)


def main():
    current = ch.read(panel.OUT)
    cue, rate = sf.read(rms.WAVE, dtype="int16")
    assert rate==16000 and ch.sha(rms.WAVE)==ch.read(rms.OUT)["sources"][rms.WAVE.name]
    small, identity = ch.load_reader()
    assert identity == current["small"]
    turbo = tx.diag.reader(current["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    builds = dict(bca=panel.middle.WORK/"build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin")==current["firmware"]
    assert tx.panel.grid.gap.uplink.firmware()==current["codec"]
    rows, guests, summary = [], [], defaultdict(Counter)
    scratch = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(scratch,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for group, i, s, old, clock, mic, assigned in cases(cue,current):
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher,native,mic)
            zero = np.zeros_like(mic)
            expect = panel.middle.pair.admit(raw,clock,panel.middle.single.hits(mic,clock),zero)
            opens = expect["opened"]
            before = bool(old and s["start"]<=old[0]<s["command_at"])
            after = bool(opens and s["start"]<=opens[0]<s["command_at"])
            rows.append(dict(group=group, scene=i, source=s, baseline_open=old,
                             result=expect, timely=after, transmit_event=clock))
            summary[group].update(trials=1, old_timely=int(before), timely=int(after),
                gained=int(after and not before), lost=int(before and not after))
            if assigned or not before or not after:
                case = (i,s,"bca","both",mic,zero,expect)
                row = tx.panel.capture(work,z,len(guests),case,small,turbo,decoder,builds,tx_pair=clock)
                guests.append(dict(row, group=group, assigned=assigned, original_timely=before, timely=after))
    assert len(rows)==225
    with ZipFile(scratch) as z:
        assert z.testzip() is None
    archive = OUT.with_suffix(".zip")
    scratch.replace(archive)
    ch.save(OUT,dict(records=rows, guest=guests, summary=summary, model=current["model"], firmware=current["firmware"],
        archive_sha256=ch.sha(archive), small=identity, turbo=current["turbo"], sources={p.name:ch.sha(p) for p in
        (rms.WAVE,rms.OUT,transfer.OUT,panel.OUT,Path(transfer.__file__),Path(__file__))},
        policy="Compare the existing equal-RMS v5 cue to original v5 under the same -6 dB panel demo, "
               "source-balanced commands and preamp gain. Verify all 225 original microphone hashes. Replay "
               "every baseline or alternate miss and each assigned weakest-gain music input on the actual P4 "
               "simulator, without further threshold or cue adjustment."))


if __name__ == "__main__":
    main()
