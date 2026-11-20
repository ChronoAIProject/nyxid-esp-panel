"""Isolate the interfering name, unaddressed command and music at the observed false wake."""
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import bella_weak_speaker_gain as trial
import check_bella_marker_only as stage

P, overlap = stage.P, stage.overlap
OUT = trial.RESULTS / "4090-c161-bella-gain-false-wake.json"


def wav(data, subtype):
    stream = BytesIO()
    sf.write(stream,data,16000,subtype=subtype,format="WAV")
    return stream.getvalue()


def main():
    plan, previous = P.read(trial.PLAN),P.read(trial.WAKE)
    assert P.sha(trial.PLAN)==previous["plan_sha256"] and P.sha(stage.GUEST)==previous["guest_sha256"]
    assert P.sha(P.QEMU)==previous["qemu_sha256"]
    meta = {r["label"]:r for r in plan["records"]}
    suspect, = [r for r in previous["records"] if meta[r["label"]]["target_id"]=="absent" and r["events"]["O"]]
    r = meta[suspect["label"]]
    assert r["mode"]=="player_x2" and r["index"]==72
    historical, = [s for s in P.read(overlap.OUT)["records"] if s["index"]==r["index"] and s["role"]=="absent"]
    grid,taps = overlap.rooms()
    case,_,scene,ref,active,tracks = next(g for g in grid[-6] if g[0]["file"]==r["file"])
    assert P.digest(ref)==r["reference_sha256"]==case["reference_sha256"]
    source,_,_,speech = next(entry for entry in overlap.event_inputs("daily") if int(entry[1])==r["index"])
    talk,level = overlap.tx.balanced.match_level(active,speech)
    assert P.digest(speech)==historical["dry_speech_sha256"] and level["gain_db"]==historical["speech_gain_db"]
    start,stop = historical["speech_start"],historical["speech_end"]
    rows = []
    with TemporaryDirectory(dir=trial.WORK) as tmp:
        work = Path(tmp)
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            def cases():
                for command, interruption, music in product((False,True),repeat=3):
                    dry = tracks["absent"].astype(np.int32)*2*int(command)
                    if interruption:
                        dry[start:stop] += 2*talk.astype(np.int32)
                    mic,clipped = scene(overlap.board.bella.rear_filter(dry,taps),include_music=music)
                    label = f"command{int(command)}-talk{int(interruption)}-music{int(music)}"
                    expected = None
                    if command and interruption and music:
                        assert P.digest(mic)==r["pcm_sha256"]
                        expected = suspect["events"]
                        for name,content in (("mic-reference.wav",wav(np.column_stack((mic,ref)),"PCM_16")),
                                             ("speaker-quarter-gain.wav",wav(dry*16384,"PCM_24"))):
                            logs.writestr(name,content)
                            v,rate=sf.read(BytesIO(content),dtype="int32" if "quarter" in name else "int16")
                            assert rate==16000 and np.array_equal(v,dry*16384 if "quarter" in name else np.column_stack((mic,ref)))
                    rows.append(dict(label=label,command=command,interruption=interruption,music=music,
                                     mic_sha256=P.digest(mic),reference_sha256=P.digest(ref),clipped=clipped))
                    yield label,mic.tobytes(),ref.tobytes(),expected,r["end"]
            replay = overlap.board.run_guest.replay_cases(stage.GUEST, Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                         P.QEMU,work,cases())
            for row,result in zip(rows,replay["records"],strict=True):
                assert row["label"]==result["label"]
                row.update(result)
                path = work/"uart"/(row["input_sha256"]+".log")
                assert P.sha(path)==row["uart_sha256"]
                logs.write(path,row["label"]+".log")
        assert len(rows)==8
        P.save(work/OUT.name,dict(index=r["index"],records=rows,source=source["text"],command=r["command"],
               command_at=r["end"],interruption_start=start,interruption_end=stop,source_sha256=r["pcm_sha256"],
               guest_sha256=replay["guest_sha256"],qemu_sha256=replay["qemu_sha256"],archive_sha256=P.sha(work/archive.name),
               sources={p.name:P.sha(p) for p in (Path(__file__),trial.PLAN,trial.WAKE,overlap.OUT)},
               policy="Factorially mute the erased-address command, competing name sentence and music before "
                      "the mic ADC. Preserve panel playback and physical R, both speaker gains, geometry and "
                      "the simulated stream clock. Stereo WAV is the exact digital M/R case. Mono quarter-gain "
                      "PCM24 is the external speaker track, with music and panel playback supplied separately."))
        (work/OUT.name).replace(OUT)
        (work/archive.name).replace(archive)


if __name__=="__main__":
    main()
