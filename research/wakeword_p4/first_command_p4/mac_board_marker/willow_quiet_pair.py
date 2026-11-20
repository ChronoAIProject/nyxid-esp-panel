"""Run the quiet case 077 as a positive playback control for both board listeners."""
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave
from zipfile import ZIP_DEFLATED, ZipFile

import run_guest
import willow_room_reference as ref

SCRIPT = Path(__file__)
ROOT = ref.RESULTS
WORK = ref.WORK.parent / "quiet"
PLAN = ROOT / "4090-hey-willow-p4-rev1-quiet-plan.json"
OUT = ROOT / "4090-hey-willow-p4-rev1-quiet.json"
PACK = ROOT / "c161-willow-077-quiet-pair.zip"
MEMBER = "first/025.wav"
GUEST = Path(".local/wakeword/c161-mac-board-marker/guest/board.bin")
SIM = Path(".local/wakeword/c161-panel-middle-uart/build")
QEMU = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")


def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    old = ref.read(ref.PLAN)
    files = {v:old["files"][v] for v in old["runtime"].values()}
    for p in (SCRIPT, ref.SCRIPT, ref.official.SOURCE, Path(ref.official.__file__), GUEST, QEMU):
        files[str(p)] = ref.digest(p.read_bytes())
    records = []
    audio = {}
    for role, pack in (("nyx","c161-bella-command-playback.zip"), ("willow","c161-bella-willow-command-playback.zip")):
        path = ROOT / pack
        files[str(path)] = ref.digest(path.read_bytes())
        with ZipFile(path) as z:
            raw = z.read(MEMBER)
            item, = (r for r in json.loads(z.read("manifest.json"))["records"] if r["file"]==MEMBER)
        assert ref.digest(raw) == item["sha256"] and item["command_at"]==44800
        audio[role] = raw
    def samples(raw):
        with wave.open(BytesIO(raw)) as wav:
            assert wav.getparams()[:3] == (1,2,16000)
            return wav.readframes(wav.getnframes())
    suffix = samples(audio["nyx"])[89600:]
    assert suffix == samples(audio["willow"])[89600:]
    buf = BytesIO()
    with wave.open(buf,"wb") as wav:
        wav.setparams((1,2,16000,0,"NONE","not compressed"))
        wav.writeframes(bytes(89600)+suffix)
    audio["erased"] = buf.getvalue()
    for role,data in audio.items():
        path = WORK / (role+".wav")
        path.write_bytes(data)
        pcm = samples(data)
        records.append(dict(role=role,path=str(path),samples=len(pcm)//2,pcm_sha256=ref.digest(pcm),sha256=ref.digest(data)))
        files[str(path)] = records[-1]["sha256"]
    plan = dict(records=records,files=files,runtime=old["runtime"],threshold_requests=["DEFAULT"],
                command_at=44800,command="turn down the hall light",source_member=MEMBER)
    if PLAN.exists():
        assert ref.read(PLAN)==plan
    else:
        ref.save(PLAN,plan)


def score(board):
    plan = ref.read(PLAN)
    cases = []
    ref_rows = []
    with TemporaryDirectory(dir=WORK) as tmp:
        folder = Path(tmp).resolve()
        exe = ref.build_feeder(board,folder)
        for i,row in enumerate(plan["records"]):
            path = Path(row["path"])
            with wave.open(str(path)) as wav:
                mic = wav.readframes(wav.getnframes())
            assert ref.digest(mic)==row["pcm_sha256"]
            cases.append((row["role"],mic,bytes(len(mic)),None,plan["command_at"]))
            run = WORK / f"run-{i}"
            rpt = ref.read(run/"report.json")
            raw = (run/"raw-returns.bin").read_bytes()
            assert ref.digest((run/"serial.log").read_bytes())==rpt["serial_sha256"]
            assert rpt["model"]=="wn9_heywillow_tts" and rpt["stage"]==104 and rpt["mode"]=="DEFAULT"
            assert rpt["source_pcm_sha256"]==row["pcm_sha256"] and ref.digest(raw)==rpt["raw_returns_sha256"]
            assert rpt["total_prime_samples"]==81600 and rpt["actual_float32_bits"]==rpt["post_prime_getter_bits"]=="3f24dd2f"
            assert not rpt["additional_priming_events"] and all(v<=0 for _,v in rpt["prime_returns"])
            pcm = mic+bytes(2*rpt["final_padding_samples"])
            returns = bytes(v+1 for _,v in rpt["prime_returns"])+raw
            result = subprocess.check_output([exe,str(len(pcm)//2),"256"],input=pcm+returns)
            events = [list(map(int,l.split())) for l in result.splitlines()]
            assert [e[0] for e in events]==[r["relative_sample"] for r in rpt["real_events"]]
            ref_rows.append(dict(model_and_read=events,uart_sha256=rpt["serial_sha256"]))
        p4 = run_guest.replay_cases(GUEST,SIM,QEMU,folder/"nyx",cases)
        history = ref.read(ROOT/"4090-c161-mac-board-marker-p4.json")
        assert all(history[k]==p4[k] for k in ("guest_sha256","qemu_sha256"))
        evidence = OUT.with_suffix(".zip")
        with ZipFile(evidence,"w",ZIP_DEFLATED) as trace, ZipFile(PACK,"w",ZIP_DEFLATED) as playback:
            for i,row in enumerate(plan["records"]):
                obs = p4["records"][i]
                assert obs["label"]==row["role"]
                trace.write(folder/"nyx/uart"/(obs["input_sha256"]+".log"),row["role"]+"/nyx.log")
                for n in ("report.json","raw-returns.bin","serial.log"):
                    trace.write(WORK/f"run-{i}"/n,row["role"]+"/willow-"+n)
                playback.write(row["path"],row["role"]+".wav")
            playback.writestr("manifest.json", json.dumps(dict(command=plan["command"],command_at=44800,
                files={r["role"]+".wav":{"sha256":r["sha256"],"pcm_sha256":r["pcm_sha256"]} for r in plan["records"]}),indent=2)+"\n")
        ref.save(OUT,dict(command_at=44800,model="C161 plus fixed marker / official Willow DEFAULT",
                 scope="Consumed quiet digital sources. Nyx on P4 QEMU, Willow on classic ESP32 and the host-compiled P4 feeder.",
                 plan_sha256=ref.digest(PLAN.read_bytes()),audio_sha256=ref.digest(PACK.read_bytes()),
                 trace_sha256=ref.digest(evidence.read_bytes()),willow=ref_rows,nyx=p4))


if __name__=="__main__":
    action = sys.argv.pop(1)
    if action=="prepare":
        setup()
    else:
        ref.PLAN,ref.WORK,ref.SCRIPT = PLAN,WORK,SCRIPT
        if action=="case":
            ref.official.PLAN,ref.official.__file__ = PLAN,str(SCRIPT)
            ref.official.case()
        elif action=="run":
            ref.run()
        else:
            score(Path(sys.argv[1]))
