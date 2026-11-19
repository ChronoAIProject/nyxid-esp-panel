"""Verify official default-threshold returns and compare full P4 Opus command captures at the same source clocks."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf
from faster_whisper import WhisperModel

import check_marker_music_commands as music
import willow_music_cases as willow
from score_handoff_willow import willow_events

ch, prior, grid = music.ch, music.prior, music.grid
OUT=ch.RESULTS/"4090-c161-willow-marker-music-p4.json"


def main():
    plan=ch.read(willow.PLAN)
    norm=ch.read(music.OUT)
    existing=ch.read(music.server.strong.OUT)
    small,identity=ch.load_reader()
    info=existing["turbo"]
    model_dir=prior.audio.prior.context.prior.audit.TURBO
    assert identity==existing["small"]==norm["small"] and all(ch.sha(model_dir/n)==h for n,h in info["files"].items())
    turbo=WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder=grid.gap.uplink.decoder_api()
    archive=OUT.with_suffix(".zip")
    tmp=archive.with_suffix(".tmp")
    rows=[]
    summary=defaultdict(Counter)
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        grid.gap.uplink.WORK=work
        for i,s in enumerate(plan["records"]):
            pcm,rate=sf.read(s["path"],dtype="int16")
            assert rate==16000 and ch.sha(Path(s["path"]))==s["sha256"] and ch.digest(pcm)==s["pcm_sha256"]
            hits=willow_events(i,s,ch.sha(willow.PLAN),z,runs=willow.WORK/"runs")
            opened=grid.gap.study.voices.timing.block(hits[0]) if hits else None
            timely=opened is not None and s["start"]<=opened<s["end"]
            row=dict(index=i,grid_index=s["grid_index"],role=s["target_id"],track=s["track"],voice=s["voice"],
                     carrier=s["carrier"],start=s["start"],command_at=s["end"],events=hits,opened=opened,timely=timely,
                     pcm_sha256=s["pcm_sha256"],small_complete=False,turbo_complete=False)
            if opened is not None and s["target_id"]!="no_address":
                cap=grid.gap.replay.capture(small,decoder,work,z,i,"willow",pcm,opened,s["carrier"])
                log=z.read(f"opus/{i:03}-willow.log").decode().replace("\r\n","\n")
                for lane,key in enumerate(("selected","raw")):
                    packets=grid.gap.uplink.packets(log,i,cap["emitted"]-cap["first"],lane)
                    wave=grid.gap.uplink.decode(decoder,packets)
                    assert ch.digest(wave)==cap["lanes"][key]["decoded_sha256"] and len(packets)==cap["lanes"][key]["packets"]
                    if lane==0:
                        large=grid.gap.uplink.recognition(turbo,wave,s["carrier"])
                row.update(capture=cap,turbo=large,small_complete=cap["lanes"]["selected"]["asr"]["complete_suffix"],
                           turbo_complete=large["complete_suffix"])
            summary[s["target_id"]].update(trials=1,opened=opened is not None,timely=timely,
                       small_complete=row["small_complete"],turbo_complete=row["turbo_complete"],
                       timely_turbo=row["turbo_complete"] and timely)
            rows.append(row)
        for p in work.glob("*-attempt*.gz"):
            z.write(p,"retry/"+p.name)
    assert len(rows)==33
    with ZipFile(tmp) as z: assert z.testzip() is None
    tmp.replace(archive)
    by_case={r["grid_index"]:r for r in plan["records"] if r["target_id"]=="native"}
    broad=[r for r in ch.read(music.study.OUT)["records"] if r["group"]=="other_music"]
    paired=[]
    for r in existing["records"]+norm["records"]:
        if r.get("local") or ("assigned" not in r.get("groups",[]) and r.get("group")!="music"):
            continue
        s=r["board"]["source"]
        i=s.get("index",s.get("grid_index"))
        # The previous recovery kept the original source metadata for all 11 assigned rows.
        original=by_case[i]
        assert s["command_at"]==original["end"] and s["carrier"]==original["carrier"]
        assert broad[i]["control_sha256"]==original["control_sha256"]
        if r["name"]=="v5" or r["name"]=="rms" or r["name"]=="peak":
            cap=r.get("uplink")
            opened=r["board"]["opened"]
            large=cap["turbo"] if cap and "turbo" in cap else (r["views"]["turbo"]["original"] if cap else {})
            paired.append(dict(grid_index=i,name=r["name"],opened=opened,small_complete=bool(cap and cap["recognition"]["complete_suffix"]),
                               turbo_complete=bool(large.get("complete_suffix")),timely=bool(opened and s["cue_at"]<=opened[0]<s["command_at"])))
    assert len(paired)==33 and all({r["name"] for r in paired if r["grid_index"]==i}=={"rms","v5","peak"} for i in by_case)
    ch.save(OUT,dict(records=rows,paired_fixed=paired,summary=summary,small=identity,turbo=info,firmware=grid.gap.uplink.firmware(),
                     archive_sha256=ch.sha(archive),sources={p.name:ch.sha(p) for p in (willow.PLAN,music.OUT,music.server.strong.OUT,Path(__file__))},
                     policy="Official ESP32 WakeNet 9 own four-voice synthetic name at native and dry RMS matched "
                     "to fixed v5, plus no-address controls. Every clip ends at the same dry command onset, "
                     "room, gain, noise, device block and scoring as the fixed-cue pair. Misses count against "
                     "first-command results. Same codec/discrete pre-roll, official detection runs on original target."))


if __name__=="__main__":
    main()
