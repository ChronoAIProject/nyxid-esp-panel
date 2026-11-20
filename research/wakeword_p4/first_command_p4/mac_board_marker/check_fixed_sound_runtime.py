"""Check a marker-only P4 receiver against archived clocks and profile both listener allocations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

from build_guest import link_guest
from check_replay import write_feeders
import run_guest
import willow_room_reference as room

HERE = Path(__file__).resolve().parent
RESULTS = room.RESULTS
SIM = Path(".local/wakeword/c161-panel-middle-uart/build").resolve()
QEMU = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32").resolve()
PRIMARY = Path(".local/wakeword/c161-mac-board-marker/src").resolve()
MODEL = PRIMARY / "models/hey-nyx-c161/model.tflite"
REPORT = RESULTS / "4090-fixed-sound-runtime-replay.json"


def digest(data):
    return sha256(data).hexdigest()


def build(board, work, audio_only, idf_python, profile_source=HERE / "profile_fixed_sound.c", *, resident_mono=False):
    work.mkdir(parents=True, exist_ok=True)
    service = board / "components/panel_services"
    runtime = board / "components/panel_wake_runtime"
    guest = work / "profile.c"
    marker = work / "marker.cpp"
    define = "#define NYX_PANEL_FIXED_SOUND 1\n" if audio_only else ""
    guest.write_text(define + ("#define NYX_PROFILE_RESIDENT_MONO 1\n" if resident_mono else "") + profile_source.read_text())
    marker.write_text(define + '#include "fixed_marker/board_marker.cpp"\n')
    sources = [("panel_voice_audio.c",guest), ("listener.cpp",marker)]
    if not audio_only:
        write_feeders(service, work)
        sources.append(("panel_wake_runtime.cpp",runtime/"panel_wake_runtime.cpp"))
    link_guest(SIM,work,"nyx_mac_listener.elf",sources,[work,service,runtime],idf_python)
    app = (work/"board.bin").read_bytes()
    assert app.count(MODEL.read_bytes()) == int(not audio_only)
    config = json.loads((SIM/"flasher_args.json").read_text())
    flash = bytearray([255])*(4<<20)
    for offset,name in config["flash_files"].items():
        data = app if name==config["app"]["file"] else (SIM/name).read_bytes()
        start = int(offset,16)
        assert start+len(data)<=2<<20
        flash[start:start+len(data)] = data
    return flash


def inputs():
    previous = room.read(room.NYX)["records"]
    for label,mic,ref_sha,at in room.inputs():
        # Read both source channels from the published room pack.
        path = room.CONTROL if label.startswith("command") else room.ROOM
        archive = room.ASSETS / room.read(path)["archive"]
        with ZipFile(archive) as z, room.wave.open(room.BytesIO(z.read("digital/"+label+".wav"))) as wav:
            data = room.np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2").reshape(-1,2)
            ref = data[:,1].tobytes()
        assert digest(ref)==ref_sha and data[:,0].tobytes()==mic
        old = previous[label]
        assert digest(run_guest.payload(mic,ref)) == old["input_sha256"]
        yield label,mic,ref,old["baseline"]["events"],old["cue_only"]["events"],at
    quiet = room.read(RESULTS/"4090-hey-willow-p4-rev1-quiet.json")
    pack = RESULTS/"c161-willow-077-quiet-pair.zip"
    assert digest(pack.read_bytes())==quiet["audio_sha256"]
    with ZipFile(pack) as z:
        for obs in quiet["nyx"]["records"]:
            name=obs["label"]
            mic=run_guest.mono(z,name+".wav")
            ref=bytes(len(mic))
            assert digest(run_guest.payload(mic,ref))==obs["input_sha256"]
            yield "quiet/"+name,mic,ref,obs["events"],None,44800
    old = room.read(RESULTS/"4090-c161-six-cups-clock.json")
    selected, = [r for r in old["records"] if r["paired_alert_removed"]]
    wav = RESULTS/"c161-six-cups-false-wake.wav"
    assert digest(wav.read_bytes())==old["wav_sha256"]
    with room.wave.open(str(wav)) as audio:
        mic=audio.readframes(audio.getnframes())
    ref=bytes(len(mic))
    assert digest(run_guest.payload(mic,ref))==selected["input_sha256"]
    yield "six-cups",mic,ref,selected["events"],None,None
    for label,mic,ref,expected,at in run_guest.cases():
        if at is None:
            yield "reference/"+label,mic,ref,expected,None,at


def boot(work, flash, cache, block, epochs):
    data = block[:4]+struct.pack("<I",epochs)+block[4:]
    flash[2<<20:(2<<20)+len(data)] = data
    image = work/"flash.bin"
    key = digest(data)
    log = cache / (key + ".log")
    if log.exists() and b"BOARD PASS\r\n" not in log.read_bytes():
        log.replace(cache / (key + ".failed-" + digest(log.read_bytes()) + ".log"))
    if not log.exists():
        image.write_bytes(flash)
        run_guest.boot(QEMU, image, log)
    output = log.read_text()
    assert all(missing not in output for missing in ("Task watchdog got triggered","Guru Meditation"))
    memories = re.findall(r"^PROFILE heap (\d+) (\w+) (\d+) (\d+)$",output,re.M)
    events = [tuple(map(int,s.split())) for s in re.findall("^PROFILE event (.*)$",output,re.M)]
    times = [tuple(map(int,s.split())) for s in re.findall("^PROFILE time (.*)$",output,re.M)]
    count, = struct.unpack_from("<I",block)
    assert len(times)==epochs and len(memories)==3*epochs
    rows=[]
    for i,(epoch,samples,size,init,total,maximum,over) in enumerate(times):
        assert i==epoch and count==samples
        heap={stage:[int(internal),int(external)] for e,stage,internal,external in memories if int(e)==i}
        assert len(heap)==3
        hits=[(m,c,o) for e,m,c,o in events if e==i]
        rows.append(dict(heap=heap,state_bytes=size,init_us=init,read_total_us=total,read_max_us=maximum,
                         reads=count//256,reads_at_least_16ms=over,
                         events=dict(M=[m for m,_,_ in hits if m],C=[c for _,c,_ in hits if c],
                                     O=[hits[0][2]] if hits else [])))
    return rows, log, sorted(cache.glob(key + ".failed-*.log"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("board",type=Path)
    ap.add_argument("work",type=Path)
    ap.add_argument("--idf-python",required=True,type=Path)
    args=ap.parse_args()
    board,work=args.board.resolve(),args.work.resolve()
    fixed=build(board,work/"fixed",True,args.idf_python.absolute())
    dual=build(PRIMARY,work/"dual",False,args.idf_python.absolute())
    current=room.read(RESULTS/"4090-c161-mac-board-marker-p4.json")
    assert digest(MODEL.read_bytes())==current["model_sha256"] and digest(QEMU.read_bytes())==current["qemu_sha256"]
    caches = {}
    for kind, flash in (("fixed", fixed), ("dual", dual)):
        prefix = sha256(flash[:2<<20]).digest()
        identity = digest(bytes.fromhex(current["qemu_sha256"]) + prefix)
        caches[kind] = work / kind / identity
        caches[kind].mkdir(exist_ok=True)
    records=[]
    shared={"quiet/nyx","quiet/erased","command1-talk1-music1","reference/self-restored-R","077/target_x2/nyx","six-cups"}
    with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as z:
        for label,mic,ref,expected,old_fixed,cmd in inputs():
            blob=run_guest.payload(mic,ref)
            row=dict(label=label,serialized_input_sha256=digest(blob),command_at=cmd,baseline=expected)
            for kind,flash in (("fixed",fixed),("dual",dual)):
                if kind=="dual" and label not in shared:
                    continue
                epochs=2 if label in shared else 1
                runs,log,failures=boot(work/kind,flash,caches[kind],blob,epochs)
                if kind=="dual":
                    target=expected
                else:
                    cue=expected["C"]
                    target=dict(M=[],C=cue,O=[(cue[0]+255)//256*256] if cue else [])
                    if old_fixed is not None:
                        assert target["O"]==old_fixed["O"] and target["C"]==old_fixed["C"]
                assert all(e["events"]==target for e in runs),(label,kind,runs,target)
                z.write(log,kind+"/"+label+".log")
                archived_failures = {}
                for attempt, failed in enumerate(failures, 1):
                    name = f"failed/{kind}/{label}-{attempt}.log"
                    z.write(failed, name)
                    archived_failures[name] = digest(failed.read_bytes())
                row[kind]=dict(epochs=runs,uart_sha256=digest(log.read_bytes()), failed_uart=archived_failures)
            records.append(row)
            print(label,flush=True)
    assert shared=={r["label"] for r in records if "dual" in r}
    trace=REPORT.with_suffix(".zip")
    (work/"uart.zip").replace(trace)
    report=dict(records=records,guest_elf_sha256={n:digest((work/n/"board.elf").read_bytes()) for n in ("dual","fixed")},
                guest_sha256={n:digest((work/n/"board.bin").read_bytes()) for n in ("dual","fixed")},
                qemu_sha256=current["qemu_sha256"],model_sha256=current["model_sha256"],
                patch_sha256=digest((HERE/"fixed-sound-runtime.patch").read_bytes()),archive_sha256=digest(trace.read_bytes()),
                scope="ESP32-P4 QEMU digital feeder, paired isolated listener allocations and guest timer observations. "
                      "Both epochs destroy and recreate the listener. The panel's physical I2S, network and call heap require board measurements.")
    room.save(REPORT,report)


if __name__=="__main__":
    main()
