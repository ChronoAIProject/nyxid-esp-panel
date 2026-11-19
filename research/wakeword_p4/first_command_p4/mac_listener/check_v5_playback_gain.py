"""Replay the amplitude-only control on the original guest and established command benchmark."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import fftconvolve

import check_repeat_command_audio as audio
import check_repeat_level as level

ch, grid, repeat = level.ch, level.grid, level.repeat
p4, gap = audio.repeat_p4, audio.gap
OUT = ch.RESULTS / "4090-c161-v5-gain-control-p4.json"
WAV = ch.RESULTS / "hye-nyx-v5-plus-0p69db.wav"


def main():
    evaluation = ch.read(level.OUT)
    previous = ch.read(repeat.OUT)
    expected = evaluation["pcm_sha256"]["boosted_v5"]
    original = grid.fit.original()[1]
    pcm = np.rint(original * evaluation["original_to_full_rms_ratio"])
    assert np.abs(pcm).max() <= 32767
    pcm = pcm.astype("<i2")
    assert ch.digest(pcm) == expected
    if not WAV.exists():
        sf.write(WAV, pcm, 16000, subtype="PCM_16")
    saved, rate = sf.read(WAV, dtype="int16")
    assert rate == 16000 and ch.digest(saved) == expected
    reader, asr = ch.load_reader()
    readings = p4.reading({"v5":original,"boosted_v5":pcm},reader)
    known = ch.read(audio.prior.OUT)
    info = known["turbo"]
    assert asr == known["small"]
    model_dir = audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / k) == h for k,h in info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = gap.uplink.decoder_api()
    selected = {i for i,(r,s) in enumerate(zip(evaluation["records"], previous["records"], strict=True))
                if r["routes"]["boosted_v5"]["external"]["timely"] and not s["routes"]["v5"]["timely"]}
    assert len(selected) == evaluation["summary"]["boosted_v5"]["gained"]
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    rows = []
    totals = defaultdict(Counter)

    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        detector = ch.binding(work / "full.so", coded="v5")
        pair = ch.binding(work / "pair.so", coded="aligned")

        def run(name, cohort, s, x, ref, want, local, kind):
            idx = len(rows)
            for attempt in range(2):
                try:
                    board = p4.phase.probe(work,s,x,ref,detector,pair,idx)
                    break
                except (AssertionError, TimeoutError):
                    logs.write(work/"serial.log", f"retry/{idx:03}-{attempt}.txt")
                    if attempt:
                        raise
            assert board["opened"] == want
            logs.writestr(f"events/{idx:03}.txt","\n".join(board["uart_events"])+"\n")
            cap = None
            if want and not local:
                cap = p4.phase.handoff.bench.fifo(work,logs,idx,x,ref,board,reader,decoder,True,True)
                decoded, packets = audio.decoded(logs,f"{idx:02}-opus.log",decoder,idx,cap["samples"],0,cap["pcm_sha256"])
                assert packets == cap["packets"]
                cap["turbo"] = gap.uplink.recognition(turbo,decoded,s["carrier"])
            totals[(cohort+"."+name+(".self" if local else ".external"))].update(
                trials=1, opened=bool(want), timely=bool(want and s["cue_at"] <= want[0] < s["command_at"]),
                small_complete=bool(cap and cap["recognition"]["complete_suffix"]),
                turbo_complete=bool(cap and cap["turbo"]["complete_suffix"]))
            rows.append(dict(index=idx,group=cohort,kind=kind,name=name,local=local,board=board,uplink=cap))
            print(idx,cohort,name,local,want,flush=True)

        index = 0
        for ids in (grid.fit.EVAL,repeat.source.EXTRA):
            for s,signals,_ in grid.scenes(ids,{"boosted_v5":pcm}):
                if index in selected:
                    new = evaluation["records"][index]
                    old = previous["records"][index]
                    assert old["source"] == new["source"] == s
                    meta = dict(cue_at=s["start"],command_at=s["command_at"],carrier=s["command"],grid_index=index)
                    for name,wave in (("v5",original),("boosted_v5",pcm)):
                        x = signals[name]
                        if name == "v5":
                            external = old["routes"]["v5"]
                            guarded = external["self"]
                        else:
                            external = new["routes"][name]["external"]
                            guarded = new["routes"][name]["original_reference"]
                        assert ch.digest(x) == external["pcm_sha256"]
                        ref = level.guard.refs(s,name,{name:wave})
                        assert ch.digest(ref) == guarded["reference_sha256"]
                        opens = [] if external["opened"] is None else [external["opened"]]
                        run(name,"gained",meta,x,np.zeros_like(x),opens,False,s["kind"])
                        run(name,"gained",meta,x,ref,guarded["opened"],True,s["kind"])
                index += 1

        for i, ((s,old,kind,h,bg,cmd),base) in enumerate(zip(audio.room.room_inputs(),known["records"],strict=True)):
            end = s["end"]
            start = end-len(pcm)
            assert (base["command_at"],base["voice"],base["command"]) == (end,s["voice"],s["carrier"])
            for name,wave in (("v5",original),("boosted_v5",pcm)):
                mix = bg+cmd
                wet = fftconvolve(wave.astype(float),h)
                mix[start:start+len(wet)] += wet
                x = audio.room.quantize(mix*s["gain"])
                if name == "v5":
                    assert ch.digest(x) == old["routes"]["fixed_v5"]["pcm_sha256"]
                meta = dict(cue_at=start,command_at=end,carrier=s["carrier"],scene=i,voice=s["voice"])
                route = grid.outcome(detector,native,x,dict(start=start,command_at=end))
                if name == "v5":
                    assert route["opened"] == base["routes"]["fixed_v5"]["opened"]
                opens = [] if route["opened"] is None else [route["opened"]]
                run(name,"common",meta,x,np.zeros_like(x),opens,False,kind)
    assert len(rows) == 4*len(selected)+32
    with ZipFile(tmp) as logs:
        assert logs.testzip() is None
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=totals,dry_readings=readings,small=asr,turbo=info,
                     gain=evaluation["original_to_full_rms_ratio"],wav_sha256=ch.sha(WAV),pcm_sha256=expected,
                     archive_sha256=ch.sha(archive),firmware_sha256=ch.sha(p4.phase.handoff.test.BUILD/"nyx_mac_listener.bin"),
                     sources={p.name:ch.sha(p) for p in (audio.prior.OUT,repeat.OUT,level.OUT,Path(__file__))},
                     policy="All gained cases, each original and boosted cue with/without playback reference. "
                     "Sixteen established room/voice/command conditions use original guest, silent reference and "
                     "full selected FIFO/Opus for both. Actual absolute SPL and board behavior need measurement."))


if __name__ == "__main__":
    native = grid.room.rooms.c161_binding()
    main()
