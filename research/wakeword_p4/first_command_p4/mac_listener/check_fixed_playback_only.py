"""Test opening only on the generated sound's marker in the digital P4 receiver."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import check_c161_event_patch_receiver as test
import check_c161_phase_agreement as phase
import check_desktop_alerts as desktop
import check_bella_marker_only_self as panel
import build_guest

base, check, overlap = test.base, test.receiver, test.overlap
run_guest = overlap.board.run_guest
PATCH = check.HERE / "fixed-playback-only.patch"
OUT = base.RESULTS / "4090-c161-fixed-playback-only.json"
SIM = Path(".local/wakeword/c161-panel-middle-uart/build")
WORK = Path(".local/wakeword/c161-fixed-playback-only")
SIX = phase.replay.six.OUT
QEMU = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")


def archived():
    rows = []
    for group, report in (("rooms", overlap.OUT), ("desktop", desktop.OUT), ("six_cups", SIX), ("panel", panel.OUT)):
        source = base.read(report)
        archive = report.with_suffix(".zip")
        assert base.sha(archive) == source["archive_sha256"]
        with ZipFile(archive) as logs:
            for r in source["records"]:
                original = r["nyx"] if group == "desktop" else r
                member = f'{r["index"]:02}/nyx.log' if group == "desktop" else r["label"] + ".log"
                uart = logs.read(member)
                assert sha256(uart).hexdigest() == original["uart_sha256"] and b"BOARD PASS" in uart
                reads = [tuple(map(int, line.split())) for line in re.findall(rb"^BOARD decision (.*)$", uart, re.M)]
                actual = dict(M=[m for m,c,o in reads if m], C=[c for m,c,o in reads if c],
                              O=[reads[0][2]] if reads else [])
                assert actual == original["events"]
                selected = [(m,c,o) for m,c,o in reads if c]
                cue = dict(M=[m for m,c,o in selected if m], C=[c for m,c,o in selected],
                           O=[selected[0][2]] if selected else [])
                if group == "desktop":
                    role = r["target_id"] + (".clean" if r["asset"]=="clean" else ".alert")
                elif group == "panel":
                    role = r["version"] + "." + r["reference"]
                else:
                    role = r.get("role", "conversation")
                rows.append(dict(group=group,role=role,label=original["label"],baseline=actual,cue_only=cue,
                                 command_at=original["command_at"],input_sha256=original["input_sha256"],
                                 uart_sha256=original["uart_sha256"]))
    return rows


def build(work, idf_python):
    # Apply the alternative to a private copy of the feeder translation unit.
    # The actual board checkout and established guest binaries are shared artifacts.
    service = Path("components/panel_services")
    root = work / "source"
    (root / service).mkdir(parents=True)
    voice = service / "panel_voice.c"
    assert base.sha(test.BOARD / voice) == base.read(overlap.board.BASE)["source_sha256"]["panel_voice.c"]
    copyfile(test.BOARD / voice, root / voice)
    subprocess.run(["patch","--batch","-p1","-i",PATCH],cwd=root,check=True)
    for rel in (service / "fixed_marker",Path("components/panel_wake_runtime"),Path("models")):
        (root / rel).symlink_to((test.BOARD / rel).resolve(),target_is_directory=True)
    assert "return decision->cue_end != 0;" in (root / voice).read_text()
    subprocess.run([sys.executable, build_guest.__file__, root, SIM, work / "guest",
                    "--idf-python", idf_python.absolute()],check=True)
    return work / "guest/board.bin"


def digital_inputs(rows, exe, records):
    known = {(r["group"],r["label"]):r for r in rows}
    plan = base.read(test.connected.PLAN)
    by_id = {r["id"]:r for r in base.read(phase.OUT)["records"] if r["cohort"]=="connected"}
    assert base.sha(test.connected.PACK) == plan["audio_sha256"]
    with ZipFile(test.connected.PACK) as pack:
        for r in plan["records"]:
            row = by_id[str(r["index"])]
            mic = overlap.study.playback.read_wav(pack,dict(file=r["member"],sha256=r["sha256"],pcm_sha256=r["pcm_sha256"]))
            assert overlap.P.digest(mic) == row["pcm_sha256"]
            mic = np.pad(mic,(0,-len(mic)%256)).tobytes()
            ref = bytes(len(mic))
            baseline = check.run(exe,mic,ref,row["baseline"],read_size=256)
            cue = check.run(exe,mic,ref,[],read_size=256)
            cue_reads = {(c-1)//256 for c in cue["C"]}
            cue["M"] = [t for t in baseline["M"] if (t-1)//256 in cue_reads]
            label = "connected/" + r["member"]
            records.append(dict(label=label,group="connected",role=r["target_id"],baseline=baseline,
                                cue_only=cue,command_at=r["end"],cue_start=r["start"]))
            yield label,mic,ref,cue,r["end"]
    room_labels = {"000/original", "000/absent", "077/original", "077/enhanced"}
    for info, role, mic, ref, _ in overlap.mixtures():
        label = f'{info["index"]:03}/{role}'
        if label in room_labels:
            r = known["rooms",label]
            records.append(dict(r,label="rooms/"+label))
            yield records[-1]["label"],mic.tobytes(),ref.tobytes(),r["cue_only"],r["command_at"]
    negative, = (r for r in base.read(SIX)["records"] if r["paired_alert_removed"])
    r = known["six_cups",negative["label"]]
    mic,rate = sf.read(phase.replay.six.WAV,dtype="int16")
    assert rate == 16000 and overlap.P.digest(mic) == negative["pcm_sha256"]
    records.append(dict(r,label="six_cups/"+r["label"]))
    yield records[-1]["label"],mic.tobytes(),bytes(mic.nbytes),r["cue_only"],None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python",type=Path,required=True)
    args = parser.parse_args()
    baseline = base.read(overlap.board.BASE)
    assert base.sha(QEMU) == baseline["qemu_sha256"] and base.sha(base.MODEL) == baseline["model_sha256"]
    rows = archived()
    WORK.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=WORK) as path:
        work = Path(path).resolve()
        guest = build(work,args.idf_python)
        assert guest.read_bytes().count(base.MODEL.read_bytes()) == 1
        exe = check.build_replay(test.BOARD.resolve(),work)
        records = []
        result = run_guest.replay_cases(guest,SIM,QEMU,work,digital_inputs(rows,exe,records))
        assert len(rows)==486 and len(records)==65
        archive = OUT.with_suffix(".zip")
        with ZipFile(work / archive.name,"w",ZIP_DEFLATED) as z:
            for row, observed in zip(records,result["records"],strict=True):
                assert row["label"]==observed["label"] and row["cue_only"]==observed["events"]
                if "input_sha256" in row:
                    assert row["input_sha256"]==observed["input_sha256"]
                row["input_sha256"] = observed["input_sha256"]
                row["cue_only_uart_sha256"] = observed["uart_sha256"]
                log = work/"uart"/(row["input_sha256"]+".log")
                assert base.sha(log)==observed["uart_sha256"]
                z.write(log,row["label"]+".log")
        counts = defaultdict(lambda:defaultdict(Counter))
        for r in rows + [r for r in records if r["group"]=="connected"]:
            for arm in ("baseline","cue_only"):
                t = r[arm]["O"]
                s = counts[r["group"]+"."+r["role"]][arm]
                s.update(cases=1,opened=int(bool(t)),changed=int(t!=r["baseline"]["O"]))
                if r["command_at"] is not None:
                    s.update(before_command=int(bool(t and r.get("cue_start",0)<=t[0]<r["command_at"])))
        sources = (Path(__file__),PATCH,overlap.OUT,desktop.OUT,SIX,panel.OUT,phase.OUT,test.connected.PLAN,
                   test.connected.PACK,overlap.board.BASE,Path(run_guest.__file__),Path(build_guest.__file__))
        base.save(work/OUT.name,dict(archived=rows,p4_replays=records,summary=counts,
                  sources={p.name:base.sha(p) for p in sources},guest_sha256=base.sha(guest),
                  model_sha256=base.MODEL_SHA,qemu_sha256=result["qemu_sha256"],archive_sha256=base.sha(work/archive.name),
                  policy="The research overlay allows a call only on the fixed cue. C161 still processes every "
                         "complete model block. 486 archived dual-route UARTs are projected at their original read "
                         "ends. The alternate P4 guest runs 60 connected clips plus five overlapping checks. "
                         "The connected baseline uses the host board feeder and authenticated frozen model clocks."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
