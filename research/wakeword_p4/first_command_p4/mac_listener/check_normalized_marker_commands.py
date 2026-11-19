"""Test both level-controlled cues on the established sixteen P4 command streams."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
from faster_whisper import WhisperModel

import check_normalized_marker_gains as test

study, ch, grid, prior = test.study, test.ch, test.grid, test.prior
OUT = ch.RESULTS/(study.fit.STEM+"-normalized-commands.json")


def main():
    dry = test.tracks()
    previous = ch.read(prior.OUT)
    reader, identity = ch.load_reader()
    info = previous["turbo"]
    model_dir = prior.audio.prior.context.prior.audit.TURBO
    assert identity==previous["small"] and all(ch.sha(model_dir/k)==h for k,h in info["files"].items())
    turbo = WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder = grid.gap.uplink.decoder_api()
    native = grid.room.rooms.c161_binding()
    observations, totals = [],defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        full = ch.binding(work/"full.so",coded="v5")
        pair = ch.binding(work/"pair.so",coded="aligned")
        for i,(s,old,kind,h,bg,cmd) in enumerate(prior.audio.room.room_inputs()):
            start = s["end"]-len(dry["v5"])
            meta = dict(scene=i,cue_at=start,command_at=s["end"],voice=s["voice"],carrier=s["carrier"])
            for name in test.levels.WAV:
                mix = bg+cmd
                wet = fftconvolve(dry[name].astype(float),h)
                mix[start:start+len(wet)] += wet
                pcm = grid.room.quantize(mix*s["gain"])
                ref = np.zeros_like(pcm)
                result = grid.outcome(full,native,pcm,dict(start=start,command_at=s["end"]))
                idx = len(observations)
                for attempt in range(2):
                    try:
                        board=prior.p4.phase.probe(work,meta,pcm,ref,full,pair,idx)
                        break
                    except (AssertionError,TimeoutError):
                        z.write(work/"serial.log",f"retry/{idx}-{attempt}.txt")
                        if attempt: raise
                want=[] if result["opened"] is None else [result["opened"]]
                assert board["opened"]==want and board["c161"]==result["c161"]
                assert board["fixed"]==[e["t"] for e in result["fixed"]]
                z.writestr(f"events/{idx:03}.txt","\n".join(board["uart_events"])+"\n")
                cap=None
                if want:
                    cap=prior.p4.phase.handoff.bench.fifo(work,z,idx,pcm,ref,board,reader,decoder,True,True)
                    wave,n=prior.audio.decoded(z,f"{idx:02}-opus.log",decoder,idx,cap["samples"],0,cap["pcm_sha256"])
                    assert n==cap["packets"]
                    cap["turbo"]=grid.gap.uplink.recognition(turbo,wave,s["carrier"])
                observations.append(dict(scene=i,kind=kind,cue=name,board=board,uplink=cap))
                totals[name].update(trials=1,timely=result["timely"],small=bool(cap and cap["recognition"]["complete_suffix"]),
                                    turbo=bool(cap and cap["turbo"]["complete_suffix"]))
                print(idx,kind,name,want,flush=True)
    with ZipFile(tmp) as z: assert len(observations)==32 and z.testzip() is None
    tmp.replace(archive)
    fw=prior.p4.phase.handoff.test.BUILD/"nyx_mac_listener.bin"
    assert ch.sha(fw)==previous["firmware_sha256"]
    ch.save(OUT,dict(records=observations,summary=totals,archive_sha256=ch.sha(archive),firmware_sha256=ch.sha(fw),
                     small=identity,turbo=info,sources={p.name:ch.sha(p) for p in (prior.OUT,test.levels.FIT,Path(__file__))},
                     policy="Both level controls frozen before playback. Original P4 simulator microphone, two-second "
                     "FIFO, original Opus encoder and full unprompted ASR. Same paired 16 benchmark cells."))


if __name__=="__main__":
    main()
