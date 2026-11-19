"""Replay every weak original v5 uplink with only its generated command raised by 6 dB."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_marked_connected_panel as mixed
import check_panel_qwen_asr as asr

tx, ch, panel = mixed.tx, mixed.ch, mixed.panel
OUT = ch.RESULTS / "4090-c161-quieter-panel-louder-weak-commands.json"


def main():
    history = ch.read(panel.OUT)
    old = {r["host_index"]:r for r in history["records"] if r["source"]["external_db"] == -6}
    older_asr = ch.read(asr.OUT)
    qwen = {r["index"]:r for r in older_asr["records"] if r["group"] == "panel.-6"}
    assert old.keys() == qwen.keys() and len(old) == 20
    proc, model, asr_id = asr.load()
    assert asr_id == older_asr["model"]
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"]/"nyx_mac_listener.bin") == history["firmware"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == history["codec"]
    small, identity = ch.load_reader()
    assert identity == history["small"]
    turbo = tx.diag.reader(history["turbo"])
    decoder = uplink.decoder_api()
    v5, _ = tx.study.panel_wave()
    quiet, sr = sf.read(panel.demo.WAVE,dtype="int16")
    assert sr == 16000 and ch.sha(panel.demo.WAVE) == history["sources"][panel.demo.WAVE.name]
    rows, summary = [], Counter()
    tmp = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        for src in ch.read(tx.diag.BOARD)["records"]:
            i,s = src["host_index"],src["source"]
            if (src["role"],src["name"]) != ("both","bca") or i not in old:
                continue
            baseline = old[i]
            assert s == baseline["source"] and qwen[i]["pcm_sha256"] == baseline["uplink"]["pcm_sha256"]
            _,bg,_,original,h,level,_ = tx.diag.scene(src)
            balanced,_ = tx.balanced.match_level(v5,original)
            boosted = mixed.master(balanced)
            (hp,_,_),_ = tx.diag.room(s["geometry"])
            local = fftconvolve(quiet.astype(float),hp)
            bg = bg.copy()
            bg[s["local_at"]:s["local_at"]+len(local)] += local
            def scene(cue,command):
                ext = fftconvolve(np.r_[cue,command].astype(float),h) * level
                mix = bg.copy()
                mix[s["start"]:s["start"]+len(ext)] += ext
                q = np.rint(mix*s["gain"])
                return q.clip(-32768,32767).astype("<i2"), int(np.count_nonzero((q < -32768) | (q > 32767)))

            prior,_ = scene(v5,balanced)
            assert ch.digest(prior) == baseline["mic_sha256"]
            pcm, clipped = scene(v5,boosted)
            np.testing.assert_array_equal(prior[:s["command_at"]],pcm[:s["command_at"]])
            clock = baseline["transmit_event"]
            zero = np.zeros_like(pcm)
            def score(x):
                raw = tx.panel.grid.fit.fixed.ch.routes(matcher,native,x)
                return panel.middle.pair.admit(raw,clock,panel.middle.single.hits(x,clock),zero)

            expected = score(pcm)
            assert expected["opened"] == baseline["opened"] and s["start"] <= expected["opened"][0] < s["command_at"]
            neg,_ = scene(np.zeros_like(v5),boosted)
            no_address = score(neg)
            k = len(rows)
            case = (i,s,"bca","both",pcm,zero,expected)
            row = tx.panel.capture(work,z,k,case,small,turbo,decoder,builds,tx_pair=clock)
            cap = row["uplink"]
            serial = z.read(f"{k:02}-opus.log").decode().replace("\r\n","\n")
            packets = uplink.packets(serial,k,cap["samples"],0)
            audio = uplink.decode(decoder,packets)
            assert len(packets) == cap["packets"] and ch.digest(audio) == cap["pcm_sha256"]
            text = asr.transcribe(proc,model,audio)
            want = asr.normalized(s["command"])
            row.update(qwen=asr.normalized(text)[-len(want):] == want, text=text, clipped_samples=clipped,
                       baseline_small=baseline["small"], baseline_turbo=baseline["turbo"], baseline_qwen=qwen[i]["complete"],
                       no_address=no_address, before=tx.balanced.dry_levels(balanced), after=tx.balanced.dry_levels(boosted))
            for tag in ("small","turbo","qwen"):
                a,b = row["baseline_"+tag],row[tag]
                summary.update({tag+"_before":int(a),tag:int(b),tag+"_gained":int(b and not a),tag+"_lost":int(a and not b)})
            summary.update(trials=1,timely=1,clipped_samples=clipped,no_address_open=int(bool(no_address["opened"])))
            rows.append(row)
            if len(rows)%5 == 0:
                print("weak",len(rows),flush=True)
    assert len(rows) == 20
    with ZipFile(tmp) as z:
        assert z.testzip() is None and sum(n.startswith("events/") for n in z.namelist()) == len(rows)
    archive = OUT.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=summary,model=history["model"],firmware=history["firmware"],codec=history["codec"],
        small=identity,turbo=history["turbo"],qwen=asr_id,archive_sha256=ch.sha(archive),sources={p.name:ch.sha(p) for p in
        (asr.OUT,panel.OUT,tx.diag.BOARD,panel.demo.WAVE,Path(mixed.__file__),Path(__file__))},
        policy="Original v5 and quiet panel on all 20 weak cases. Restore each baseline microphone and verify "
        "identical samples through command onset. Raise only the previously balanced prerecorded command "
        "by 6 dB using the same smoothed -1 dBFS limiter as the marked sentences. Preserve scene gains. "
        "Complete selected Opus on real simulated openings, paired recognizer scores and host no-address controls."))


if __name__ == "__main__":
    main()
