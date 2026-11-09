#!/usr/bin/env python3
"""Run offline PCM through Hi ESP on a verified development board, then restore.

Only the explicitly verified active app slot is written. No bootloader,
partition table, NVS or filesystem is flashed. Keep outputs outside git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import wave
import zlib

import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--expected-mac", required=True)
    parser.add_argument("--status-url", required=True)
    parser.add_argument("--restore-app", type=Path, required=True)
    parser.add_argument("--expected-app-sha256", required=True, help="Original app ELF SHA from /status")
    parser.add_argument("--app-offset", type=lambda s: int(s,0), required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--slot-bytes", type=lambda s: int(s,0), required=True)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baud", type=int, choices=(115200,921600,1500000), default=1500000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"report.json").exists():
        parser.error("Use a fresh output directory")
    if not args.status_url.startswith("http://"):
        parser.error("Status URL must explicitly name the development board's LAN HTTP endpoint")

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def esptool(arguments, log, timeout=120):
        command = [sys.executable,"-m","esptool","--chip","esp32p4","--port",args.port,*arguments]
        value = subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        (args.output/log).write_text(value.stdout+value.stderr)
        value.check_returncode()
        return value.stdout

    def flash(path, log):
        esptool(["--baud","1500000","--before","default_reset","--after","hard_reset",
                 "write_flash",hex(args.app_offset),str(path)],log)

    before = status()
    if before["app_sha256"] != args.expected_app_sha256 or before["ota"]["running"] != args.slot:
        raise RuntimeError("Original app or active slot does not match the explicit development-board expectation")
    if before["voice"]["active"] or before["ota"]["busy"]:
        raise RuntimeError("Development board is busy")
    if bytes.fromhex(args.expected_app_sha256) not in args.restore_app.read_bytes():
        raise RuntimeError("Restore binary does not contain the expected ELF identity")
    if not all(0 < path.stat().st_size <= args.slot_bytes for path in (args.probe_bin,args.restore_app)):
        raise RuntimeError("Probe or restore app exceeds the explicitly verified slot size")
    rows = list(map(json.loads,args.manifest.read_text().splitlines()))
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        path = args.manifest.parent/row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Fixture WAV hash mismatch")
    chip = esptool(["chip_id"],"chip-private.log",30)
    if args.expected_mac.lower() not in chip.lower():
        raise RuntimeError("USB MAC does not match the development board")
    report = {"probe_sha256":hashlib.sha256(args.probe_bin.read_bytes()).hexdigest(),
              "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "engine":"wn9_hiesp", "mode":"DET_MODE_95", "sample_rate":16000,
              "scope":"Offline PCM detector benchmark, not acoustic/full-app acceptance.", "clips":[]}
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Benchmark interrupted; restoring original app")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        flash(args.probe_bin,"probe-flash-private.log")
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=30) as uart, (args.output/"uart-private.log").open("w") as log:
            uart.dtr=False;uart.rts=True;time.sleep(.1);uart.rts=False
            uart.reset_input_buffer()
            def message(prefix, timeout=30):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    line=uart.read_until(b"\n",8192).decode("utf-8",errors="replace").strip()
                    if line:
                        log.write(line+"\n");log.flush()
                    marker=line.find(prefix+" ")
                    # Boot console reconfiguration can leave a partial log
                    # before the first frame. Framing is by our unique marker,
                    # with id/length/CRC checks before accepting PCM results.
                    if marker>=0:
                        return json.loads(line[marker+len(prefix)+1:])
                    if "WAKE_BENCH_ERROR" in line or "abort()" in line or "Guru Meditation" in line:
                        raise RuntimeError("Benchmark firmware reported an error; see private UART log")
                raise TimeoutError(prefix)
            ready=message("WAKE_BENCH_READY",45)
            assert ready["rate"] == 16000
            if args.baud!=115200:
                uart.write(f"BAUD {args.baud}\n".encode("ascii"))
                assert message("WAKE_BENCH_BAUD")["baud"]==args.baud
                uart.baudrate=args.baud
                ready=message("WAKE_BENCH_READY",5)
                assert ready["rate"]==16000
            for index,row in enumerate(rows):
                with wave.open(str(args.manifest.parent/row["path"]),"rb") as wav:
                    if (wav.getframerate()!=16000 or wav.getnchannels()!=1 or wav.getsampwidth()!=2
                            or wav.getnframes()!=row["samples"] or wav.getcomptype()!="NONE"):
                        raise ValueError("Expected exact 16 kHz mono PCM16 fixture")
                    samples=wav.getnframes()
                    data=wav.readframes(samples)
                    if len(data)!=samples*2:
                        raise ValueError("Truncated WAV fixture")
                uart.write(f"CLIP {index} {samples}\n".encode("ascii"))
                prepared=message("WAKE_BENCH_CLIP_READY")
                assert prepared["id"] == index
                for at in range(0,len(data),16384):
                    part=data[at:at+16384]
                    if uart.write(part)!=len(part):
                        raise RuntimeError("Short UART write")
                uart.flush()
                result=message("WAKE_BENCH_RESULT",max(30,samples/16000))
                if not (result["id"]==index and result["complete"] and result["crc32"]==zlib.crc32(data)):
                    raise RuntimeError("PCM transfer verification failed")
                report["clips"].append({**row,"result":result,"threshold":prepared["threshold"]})
                (args.output/"progress.json").write_text(json.dumps(report,indent=2)+"\n")
                if index%10==0 or index==len(rows)-1:
                    print(json.dumps({"phase":"offline_hiesp","completed":index+1,"total":len(rows)}),flush=True)
    finally:
        # First close UART, then restore even after a failed flash, transfer or
        # assertion. SIGTERM becomes an exception so normal cancellation also
        # enters this path. SIGKILL/power removal cannot be handled in software.
        print("Restoring original development app",flush=True)
        flash(args.restore_app,"restore-private.log")
        deadline=time.monotonic()+60
        after=None
        while time.monotonic()<deadline:
            try:
                after=status()
                if (after["app_sha256"]==args.expected_app_sha256 and after["nyx"]["state"]=="up"
                        and after["voice"]["wake"]["listening"]):
                    break
            except (OSError,ValueError):
                pass
            time.sleep(1)
        if not after or after["app_sha256"]!=args.expected_app_sha256 or after["ota"]["running"]!=args.slot:
            raise RuntimeError("RESTORE VERIFICATION FAILED")
        if after["nyx"]["state"]!="up" or not after["voice"]["wake"]["listening"]:
            raise RuntimeError("Original app restored, but local service/wake not ready")
        if after["voice"]["service_id"]!=before["voice"]["service_id"]:
            raise RuntimeError("Original voice service selection changed")
        report["restored"]={"app_sha256":after["app_sha256"],"nyx_state":after["nyx"]["state"],"wake_listening":True}
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("Original app, local connection and Hi ESP listening verified",flush=True)


if __name__ == "__main__":
    main()
