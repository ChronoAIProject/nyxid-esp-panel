"""Package paired speaker levels and clocked digital missing-reference probes actually run on P4."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import check_panel_echo_confirmation as echo
import check_panel_middle_p4 as middle
import check_panel_skew_and_bot as skew
import package_timed_panel as previous
from package_context_playback import add, formatted
from compare_room_willow_uplink import put_audio

ch, tx = echo.ch, echo.tx
PACK = ch.RESULTS/"c161-panel-level-reference-mac.zip"


def main():
    timed = ch.read(tx.BOARD)
    confirmed = ch.read(echo.BOARD)
    shifted = ch.read(skew.OUT)
    strong_c = ch.read(middle.OUT)
    c_rows = {(r["group"],r["host_index"]):r for r in strong_c["records"]}
    balanced = ch.read(tx.balanced.OUT)
    original = ch.read(tx.diag.BOARD)
    base_rows = {r["index"]:r for r in original["records"] if r["name"]=="bca" and r["role"]=="both"}
    matched = {r["baseline_index"]:r for r in balanced["records"] if r["mode"]=="balanced"}
    confirm_rows = {(r["group"],r["host_index"]):r for r in confirmed["records"]}
    timer_rows = {(r["group"],r["host_index"]):r for r in timed["records"]}
    v5, bca = tx.study.panel_wave()
    source_bank = tx.diag.sources()[-1]
    tmp = PACK.with_suffix(".tmp")
    written, cases, speakers = {}, [], []
    with ZipFile(tmp,"w") as z:
        for path, report in ((tx.BOARD,timed),(echo.BOARD,confirmed),(middle.OUT,strong_c),(skew.OUT,shifted),(tx.balanced.OUT,balanced)):
            assert ch.sha(path.with_suffix(".zip"))==report["archive_sha256"]

        def wave(name,pcm):
            digest = ch.digest(pcm)
            if digest not in written:
                written[digest] = put_audio(z,name,pcm)["file"]
            return written[digest]

        def probe(label, mic, ref, row, report):
            assert (ch.digest(mic),ch.digest(ref))==(row["mic_sha256"],row["reference_sha256"])
            a = report.with_suffix(".zip")
            with ZipFile(a) as raw:
                uart = raw.read(f'events/{row["index"]:03}.txt')
            prefix = f'digital/{label}'
            add(z,prefix+".txt",uart)
            cases.append(dict(label=label,m=wave(prefix+"-M.wav",mic),r=wave(prefix+"-R.wav",ref),
                              uart=prefix+".txt",source=row["source"],firmware=report.name,index=row["index"],
                              tx_capture_sample=row.get("transmit_event",0),opened=row["opened"],
                              small=row["small"],turbo=row["turbo"]))

        panel_wav = wave("speakers/panel-demo.wav",bca)
        for idx,r in base_rows.items():
            s = r["source"]
            if s["offset"] or s["external_db"]!=-6:
                continue
            g = s["geometry"]
            command = source_bank[s["voice"],s["command"]]
            up, stats = tx.balanced.match_level(v5,command)
            old = wave(f"speakers/g{g}-original.wav",np.r_[v5,command])
            new = wave(f"speakers/g{g}-balanced.wav",np.r_[v5,up])
            speakers.append(dict(geometry=g,voice=s["voice"],command=s["command"],panel=panel_wav,
                                 original=old,balanced=new,levels=stats,cue_samples=len(v5)))
            _,mic,ref,_,_ = next(tx.balanced.inputs(r))
            probe(f"g{g}-balanced-real-R",mic,ref,matched[idx],tx.balanced.OUT)
            probe(f"g{g}-balanced-zero-R",mic,np.zeros_like(ref),c_rows["balanced.-6",r["host_index"]],middle.OUT)
        # Include an observed timer-only loss and the matching confirmed result on identical natural speech.
        for i,(_,mic) in enumerate(tx.spoken_inputs()):
            if ("natural.with_event",i) not in timer_rows: continue
            a,b = timer_rows["natural.with_event",i],confirm_rows["isolated_spoken",i]
            if not a["opened"] and b["opened"]:
                ref=np.zeros_like(mic)
                probe("natural-timer",mic,ref,a,tx.BOARD)
                probe("natural-confirmed",mic,ref,b,echo.BOARD)
                probe("natural-middle",mic,ref,c_rows["isolated_spoken",i],middle.OUT)
                break
        for i,(_,mic,_,_) in enumerate(echo.room_trials()):
            for kind in ("air_conditioner","music"):
                row=confirm_rows.get(("room."+kind,i))
                if row and row["opened"]:
                    probe(f"residual-room-{i}-pair",mic,np.zeros_like(mic),row,echo.BOARD)
                    probe(f"residual-room-{i}-middle",mic,np.zeros_like(mic),c_rows["room_residual",i],middle.OUT)
        # This exact original smoke fixture appears in both runs with no TX event.
        _,_,_,mic,ref,_,_=tx.single.false_alert()
        probe("across",mic,ref,c_rows["across",0],middle.OUT)
        # Include one previously self-opening, now suppressed panel-only playback.
        for i,(_,_,_,mic,_) in enumerate(tx.study.simultaneous()):
            row=c_rows.get(("panel",i))
            if row and not row["opened"]:
                probe("self-zero-R",mic,np.zeros_like(mic),row,middle.OUT)
                break
        by_source = {r["host_index"]:r for r in shifted["records"] if r["group"]=="bot"}
        for (meta,pcm,_,_,_),old in zip(skew.everyday.sources("bot"),by_source.values(),strict=True):
            s = old["source"]
            assert ch.digest(pcm)==s["full_pcm_sha256"]
            begin=s["source_offset"]
            clipped=pcm[begin:min(s["spans"][-1]["end"]+16000,len(pcm))].copy()
            probe(f'bot-{old["host_index"]}',clipped,np.zeros_like(clipped),old,skew.OUT)
        chosen = [r for r in shifted["records"] if r["group"]=="panel" and r["delta"] in (-1280,-256,1280)]
        first,= {r["host_index"] for r in chosen}
        _,_,_,mic,_=next(case for i,case in enumerate(tx.study.simultaneous()) if i==first)
        for r in chosen:
            probe(f'panel-skew-{r["delta"]}',mic,np.zeros_like(mic),r,skew.OUT)
        for f in (tx.HEADER,echo.HEADER,middle.HEADER,tx.single.timed.HEADER,tx.study.COEFF):
            add(z,"receiver/"+f.name,f.read_bytes())
        add(z,"receiver/panel_marker_pair.h",tx.study.reference_header().encode())
        assert len(speakers)==4 and any(c["label"]=="natural-confirmed" for c in cases)
        files={n:dict(bytes=len(z.read(n)),sha256=sha256(z.read(n)).hexdigest()) for n in z.namelist()}
        for digest, name in written.items(): files[name]["pcm_sha256"]=digest
        docs=(tx.diag.BOARD,tx.balanced.OUT,tx.BOARD,echo.BOARD,middle.OUT,skew.OUT,Path(__file__))
        add(z,"manifest.json",formatted(dict(files=files,speakers=speakers,cases=cases,
            model_sha256=confirmed["model"],threshold=0.91796875,smoothing_steps=5,
            firmware={tx.balanced.OUT.name:original["versions"]["panel_order"],tx.BOARD.name:timed["firmware"],
                      echo.BOARD.name:confirmed["firmware"],middle.OUT.name:strong_c["firmware"],skew.OUT.name:shifted["firmware"]},
            sources={p.name:ch.sha(p) for p in docs},
            speech_excerpt="Earnings-22, Rev.com speech-datasets, recording 4483680. Eight-second diagnostic excerpt, 16 kHz mono. Source URL in previous c161-timed-panel-mac-ab.zip manifest.",
            format="16 kHz mono PCM16. Digital M and R start at the same capture sample. The smoke runs prime "
                   "C161 with 81600 samples. tx_capture_sample is 1-based and zero means no local event.",
            replay="Speaker A/B: set the same positions and hardware gain and play the panel demo plus one "
                   "external track simultaneously. Each pair has the same v5 prefix and the same command "
                   "start; only the sender's command gain differs. Digital A/B: for balanced-real-R use the "
                   "previous timed+BCA firmware; for missing R test the firmware identified in each row. "
                   "Timer uses PanelDemoGuard, confirmed uses PanelEchoConfirmation plus independent B,C matchers "
                   "on R and M. The middle-marker variant uses PanelEchoConfirmation with PanelMarkerPair on R and PanelMiddleMarker on M. Feed the actual R to the AEC/FIFO. The separately dated playout event is the "
                   "B,C match in the panel demo at dry sample 14388, mapped to the capture clock, on three reproductions the single-marker guard worked at -16, 0, +16 and +80 ms; "
                   "at -80 ms it missed all three panel returns. A queued playback request alone cannot establish this clock. "
                   "Record actual M/R, playout and OPEN clocks, first packet, decoded command, slowest block "
                   "and minimum internal heap on both boards.")))
    previous.verify(tmp)
    tmp.replace(PACK)
    print(PACK.stat().st_size,ch.sha(PACK))


if __name__=="__main__":
    main()
