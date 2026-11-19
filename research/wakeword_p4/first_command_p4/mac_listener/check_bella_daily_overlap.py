"""Replay original and louder fixed calls while another synthetic speaker talks over the address."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_room_willow as matched
import check_bella_board_reference as board
import check_bella_peak_playback as peak
import check_eight_voice_self_play as dry
from check_c161_event_transcripts_v1 import event_inputs

P, tx, study = board.ch, board.tx, board.study
WORK = Path(".local/wakeword/c161-bella-daily-overlap")
OUT = P.RESULTS / "4090-c161-bella-daily-overlap.json"
DAILY = P.RESULTS / "4090-c161-stream-stretch-daily-v1.json"


def rooms():
    grid = P.read(matched.PLAN)
    captures = P.read(board.bella.OUT)["captures"]
    original = P.read(board.OUT)
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    assert rate == 16000
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    filters, _ = board.bella.gains.direction.filters()
    cue, rate = sf.read(board.bella.CUE,dtype="int16")
    louder, rate2 = sf.read(peak.CUE,dtype="int16")
    gain = P.read(peak.OUT)
    assert rate == rate2 == 16000 and len(cue) == len(louder)
    assert P.sha(peak.CUE) == gain["cue_sha256"] and P.sha(board.bella.CUE) == gain["input_cue_sha256"]
    assert P.sha(board.weak.PACK) == board.run_guest.BELLA_SHA
    rooms_by_db = defaultdict(list)
    with ZipFile(board.weak.PACK) as pack:
        entries = {r["file"]:r for r in json.loads(pack.read("manifest.json"))["records"]}
        for i,(case, cap) in enumerate(zip(grid["cases"],captures,strict=True)):
            s = cap["source"]
            entry = entries[case["file"]]
            wave = study.playback.read_wav(pack,entry)
            end = s["command_at"]
            assert P.digest(wave[end:]) == entry["command_pcm_sha256"]
            assert np.array_equal(wave[end-len(cue):end],cue)
            active_cue = wave[s["start"]:end].copy()
            assert tx.balanced.dry_levels(active_cue) == grid["levels"]["nyx_original"]
            absent = wave.copy()
            absent[:end] = 0
            enhanced = wave.copy()
            enhanced[end-len(louder):end] = louder
            _, (hx, _, _) = tx.diag.room(s["geometry"])
            n = case["samples"]-4800-len(hx)+1
            scene = study.mixed.room_scene(s,{"length":(entry,range(n))},panel,v5,music,music_id)
            reference = np.zeros(case["samples"],dtype="<i2")
            reference[s["panel_at"]:s["panel_at"]+len(panel)] = panel
            assert P.digest(reference) == case["reference_sha256"]
            clean, _ = scene(board.bella.rear_filter(wave,filters["back"]))
            erased, _ = scene(board.bella.rear_filter(absent,filters["back"]))
            assert P.digest(clean) == case["nyx_mic_sha256"] and P.digest(erased) == case["no_address_mic_sha256"]
            assert original["records"][2*i]["events"]["O"] and not original["records"][2*i+1]["events"]["O"]
            rooms_by_db[s["external_db"]].append((case,s,scene,reference,active_cue,dict(original=wave,enhanced=enhanced,absent=absent)))
    assert {k:len(v) for k,v in rooms_by_db.items()} == {-6:8,0:8,6:8}
    return rooms_by_db, filters["back"]


def mixtures():
    grid, back = rooms()
    clean = [r for r in P.read(dry.OUT)["ordinary"] if r["group"]=="daily"]
    phrases = defaultdict(list)
    for i,((row,key,events,speech), witness) in enumerate(zip(event_inputs("daily"),clean,strict=True)):
        assert witness["index"] == i and row["key"] == key and row["pcm_sha256"] == P.digest(speech)
        assert row["pcm_sha256"] == witness["result"]["pcm_sha256"] and not witness["result"]["opened"]
        phrases[row["text"]].append((i,row,speech))
    assert len(phrases) == 32 and all(len(v)==3 for v in phrases.values())
    for j, variants in enumerate(phrases.values()):
        # Rotate renditions through levels. Every text visits each level, every room is used four times.
        for k,(i,row,speech) in enumerate(variants):
            db = (-6,0,6)[(j+k)%3]
            case,s,scene,ref,active,tracks = grid[db][j%8]
            talk, levels = tx.balanced.match_level(active,speech)
            start = (s["start"]+s["command_at"]-len(talk))//2
            stop = start+len(talk)
            assert 0 <= start and stop <= len(tracks["original"])
            info = dict(index=i,group=row["group"],text=row["text"],voice=row["voice"],
                        dry_speech_sha256=row["pcm_sha256"],speech_gain_db=levels["gain_db"],
                        speech_active_dbfs=levels["after"]["active_rms_dbfs"],speech_start=start,speech_end=stop,
                        file=case["file"],db=db,cue_start=s["start"],command_at=s["command_at"],command=s["command"],
                        reference_sha256=case["reference_sha256"],samples=len(ref))
            for role, dry_wave in tracks.items():
                # Superpose sources before the microphone ADC. Only the panel goes to electrical R.
                external = dry_wave.astype(float)
                external[start:stop] += talk
                mic, clips = scene(board.bella.rear_filter(external,back))
                assert len(mic) == len(ref) and (len(mic)+255)//256*256 <= 160000
                yield info, role, mic, ref, clips


def main():
    expected = P.read(board.BASE)
    guest = Path(".local/wakeword/c161-mac-board-marker/guest/board.bin")
    sim = Path(".local/wakeword/c161-panel-middle-uart/build")
    qemu = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")
    assert P.sha(guest) == expected["guest_sha256"] and P.sha(qemu) == expected["qemu_sha256"]
    WORK.mkdir(exist_ok=True,parents=True)
    rows = []
    def cases():
        for info,role,mic,ref,clips in mixtures():
            label = f'{info["index"]:03}/{role}'
            rows.append(dict(info,role=role,label=label,mic_sha256=P.digest(mic),clipped=clips))
            yield label,mic.tobytes(),ref.tobytes(),None,info["command_at"]
            if len(rows)%48 == 0:
                print(len(rows),"of 288",flush=True)
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        result = board.run_guest.replay_cases(guest,sim,qemu,work,cases())
        counts = defaultdict(Counter)
        for r,test in zip(rows,result["records"],strict=True):
            assert r["label"] == test["label"]
            r.update(test)
            opened = test["events"]["O"]
            timely = bool(opened and r["cue_start"] <= opened[0] < r["command_at"])
            for subset in ("all",str(r["db"]),r["group"]):
                counts[r["role"]+"."+subset].update(cases=1,opened=bool(opened),timely=timely,
                    early=bool(opened and opened[0]<r["cue_start"]),late=bool(opened and opened[0]>=r["command_at"]),
                    allowed_model=bool(test["events"]["M"]),allowed_marker=bool(test["events"]["C"]))
        assert len(rows)==len({r["input_sha256"] for r in rows})==288
        assert all(n==4 for n in Counter(r["file"] for r in rows if r["role"]=="original").values())
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for r in rows:
                path = work/"uart"/(r["input_sha256"]+".log")
                assert P.sha(path) == r["uart_sha256"]
                z.write(path,r["label"]+".log")
        P.save(work/OUT.name,dict(records=rows,summary=counts,model=expected["model_sha256"],
               guest_sha256=result["guest_sha256"],qemu_sha256=result["qemu_sha256"],archive_sha256=P.sha(work/archive.name),
               sources={p.name:P.sha(p) for p in (Path(__file__),matched.PLAN,board.OUT,board.BASE,board.bella.OUT,
                       board.bella.CUE,peak.CUE,peak.OUT,board.weak.PACK,dry.OUT,DAILY)},
               policy="32 consumed synthetic everyday texts, three renditions each from one voice. Rotate the "
                      "renditions through the three existing playback levels in the 24 rear-room scenes. Center each "
                      "whole everyday file on the fixed address interval, with the old active-RMS level rule and "
                      "minus-one dBFS peak cap. Two external sources share the same rear acoustic transfer. Music and "
                      "panel playback retained, physical R contains only the panel. Pair original, +5.68 dB and erased "
                      "address with the same speech and command. Separate cold boot and full input per arm. Development "
                      "stress cases share contexts and contribute no independent background hours."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print({k:dict(v) for k,v in counts.items() if k.endswith(".all")})


if __name__ == "__main__":
    main()
