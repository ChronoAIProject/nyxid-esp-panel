"""Balance a generated command against the fixed external cue and replay the actual combined P4 receiver."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import fftconvolve

import diagnose_panel_command_levels as diag
from check_c161_command_playback_level import levels as dry_levels
import check_timed_final_marker as timed

panel, study, ch = diag.panel, diag.study, diag.ch
OUT = ch.RESULTS / "4090-c161-panel-balanced-playback.json"
BUILD = panel.WORK.with_name(panel.WORK.name + "-timed") / "build"


def match_level(cue, command):
    target = dry_levels(cue)
    before = dry_levels(command)
    wanted = target["active_rms_dbfs"] - before["active_rms_dbfs"]
    gain_db = min(wanted, -1.0 - before["peak_dbfs"])
    x = np.rint(command.astype(float)*10**(gain_db/20)).astype("<i2")
    return x, dict(target=target,before=before,after=dry_levels(x),gain_db=float(gain_db))


def inputs(row):
    s = row["source"]
    original, noise, local, command, hx, gain, _ = diag.scene(row)
    cue, bca = study.panel_wave()
    adjusted, levels = match_level(cue,command)
    at, local_at, end = s["start"], s["local_at"], s["command_at"]
    ref = np.zeros_like(original)
    ref[local_at:local_at+len(bca)] = bca
    assert ch.digest(ref) == row["reference_sha256"]
    base = noise.copy()
    base[local_at:local_at+len(local)] += local
    for label, dry in (("balanced",np.r_[cue,adjusted]),("no_external_cue",np.r_[np.zeros_like(cue),adjusted])):
        mic = base.copy()
        wet = fftconvolve(dry.astype(float),hx)*gain
        mic[at:at+len(wet)] += wet
        mic *= s["gain"]
        clipped = int(np.count_nonzero((np.rint(mic)<-32768) | (np.rint(mic)>32767)))
        pcm = np.rint(mic).clip(-32768,32767).astype("<i2")
        if label == "balanced":
            np.testing.assert_array_equal(pcm[:end],original[:end])
        yield label, pcm, ref, levels, clipped


def main():
    report = ch.read(diag.BOARD)
    versions = report["versions"]
    assert ch.sha(BUILD/"nyx_mac_listener.bin") == versions["panel_order"]
    assert panel.grid.gap.uplink.firmware() == versions["codec"]
    small, small_info = ch.load_reader()
    assert small_info == versions["small"]
    turbo = diag.reader(versions["turbo"])
    decoder = panel.grid.gap.uplink.decoder_api()
    rows, summary = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        det = (ch.binding(work/"strict.so",coded="v5"),timed.binding(work),ch.binding(work/"prefix.so",coded="aligned"),
               study.reference_binding(work/"panel"),panel.grid.room.rooms.c161_binding())
        for before in report["records"]:
            if (before["name"],before["role"]) != ("bca","both"):
                continue
            s = before["source"]
            for mode,pcm,ref,levels,clipped in inputs(before):
                idx = len(rows)
                host = study.score(pcm,ref,det,{})
                case = (before["host_index"],s,"bca","both",pcm,ref,host)
                row = panel.capture(work,z,idx,case,small,turbo,decoder,dict(bca=BUILD))
                row.update(mode=mode,baseline_index=before["index"],baseline_opened=before["opened"],
                           baseline_small=before["small"],baseline_turbo=before["turbo"],source_levels=levels,clipped_samples=clipped)
                if mode=="balanced" and before["opened"] and before["opened"][0]<s["command_at"]:
                    assert row["opened"]==before["opened"]
                timely = bool(row["opened"] and s["start"] <= row["opened"][0] < s["command_at"])
                stats = summary[f'{mode}.{s["external_db"]}']
                stats.update(trials=1,opens=int(bool(row["opened"])),timely=int(timely),
                             small=int(row["small"]),turbo=int(row["turbo"]))
                if mode=="balanced":
                    for key in ("small","turbo"):
                        stats.update({key+"_baseline":int(before[key]),key+"_gained":int(row[key] and not before[key]),
                                      key+"_lost":int(before[key] and not row[key])})
                rows.append(row)
                print(idx,before["index"],mode,row["opened"],row["turbo"],flush=True)
    assert len(rows)==120 and all(v["trials"]==20 for v in summary.values())
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=summary,versions=versions,archive_sha256=ch.sha(archive),sources={
        p.name:ch.sha(p) for p in (diag.BOARD,Path(diag.__file__),Path(__file__))},
        policy="Pair all 60 simultaneous-source development scenes for the timed receiver and reordered panel. "
               "Scale only the dry command to the original cue's active RMS (20 ms frames within 30 dB of the "
               "strongest frame); limit linear gain to -1 dBFS peak. Restore original physical gains including "
               "the music and panel, saturating only at PCM16 bounds and counting clipped samples. Reproduce the original mic SHA and verify all pre-command samples. Repeat "
               "on P4 with only the external cue zeroed and the balanced command retained. Unopened trials count "
               "as incomplete. A sender-side change to generated fixtures."))


if __name__ == "__main__":
    main()
