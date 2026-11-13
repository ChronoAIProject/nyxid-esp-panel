#!/usr/bin/env python3
"""Run offline PCM through a selected reference on a verified P4, then restore.

Only the explicitly verified active app slot is written. No bootloader,
partition table, NVS or filesystem is flashed. Keep outputs outside git.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import wave
import zlib

import serial


CONTROL_PROTOCOL = "official-threshold-v1"
CONTROL_FIELD = "reference_threshold_f32_hex"


def binary32_value(bits):
    if not isinstance(bits,str) or not re.fullmatch("[0-9a-f]{8}",bits):
        raise ValueError("Expected eight lowercase binary32 hex digits")
    value=struct.unpack(">f",bytes.fromhex(bits))[0]
    if not math.isfinite(value):
        raise ValueError("Threshold must be finite")
    return value


def request_value(bits):
    if bits is None:
        return None
    value=binary32_value(bits)
    minimum=struct.unpack(">f",struct.pack(">f",.4))[0]
    maximum=struct.unpack(">f",struct.pack(">f",.9999))[0]
    if not minimum<=value<=maximum:
        raise ValueError("Threshold binary32 request is outside the official API range")
    return value


def threshold_command(row,index,enabled):
    if (CONTROL_FIELD in row)!=enabled:
        raise ValueError("Threshold request fields and explicit --threshold-control must agree")
    if not enabled:
        return f"CLIP {index} {row['samples']}\n"
    value=request_value(row[CONTROL_FIELD])
    if type(row['samples']) is not int or not 0<row['samples']<=16000*3600:
        raise ValueError("Invalid complete-stream sample count")
    suffix="DEFAULT" if value is None else "SET "+row[CONTROL_FIELD]
    return f"CLIP {index} {row['samples']} {suffix}\n"


def validate_control_initialization(prepared,row,index):
    control=prepared["control"]
    if prepared["id"]!=index or prepared["chunk"]!=512 or control["schema"]!=CONTROL_PROTOCOL:
        raise ValueError("Unexpected controlled reference identity/chunk")
    bits=row[CONTROL_FIELD]
    request_value(bits)
    if (control["requested_f32_hex"]!=bits or control["setter_called"] is not (bits is not None)
            or control["setter_result"]!=(1 if bits is not None else None)):
        raise ValueError("Reference setter did not match the explicit binary32 request")
    for value_key,bits_key in (("default_getter","default_getter_f32_hex"),
                               ("getter_after_set","getter_after_set_f32_hex")):
        observed=control[bits_key]
        if not 0<binary32_value(observed)<1:
            raise ValueError("Getter is outside the finite probability interval")
        value=control[value_key]
        if not math.isfinite(value) or struct.pack(">f",value).hex()!=observed:
            raise ValueError("Getter decimal and binary32 identity differ")
    observed=control["getter_after_prime_f32_hex"]
    if not 0<binary32_value(observed)<1:
        raise ValueError("Post-prime getter is outside the finite probability interval")
    if not math.isfinite(prepared["threshold"]) or struct.pack(">f",prepared["threshold"]).hex()!=observed:
        raise ValueError("Post-prime getter identity differs")
    if bits is None and control["default_getter_f32_hex"]!=control["getter_after_set_f32_hex"]:
        raise ValueError("DEFAULT path changed getter without detection or a setter")
    prime=control["prime_returns"]
    if len(prime)!=63 or any(type(value) is not int or value not in (-1,0,1) for value in prime):
        raise ValueError("Incomplete or invalid raw priming return stream")


def validate_control_result(result,events,row,index,chunk):
    calls=(row["samples"]+chunk-1)//chunk
    if (result["id"]!=index or result["complete"] is not True or result["samples"]!=row["samples"]
            or result["calls"]!=calls or result["events"]!=len(events)
            or result["events_emitted"]!=len(events)
            or result["offsets_complete"] is not (len(events)<=64)
            or result["no_detect_calls"]+result["channel_verified_calls"]+len(events)!=calls
            or min(result["no_detect_calls"],result["channel_verified_calls"])<0):
        raise ValueError("Incomplete raw reference event/call accounting")
    previous=0
    for event in events:
        call=event["call"]
        if (event["id"]!=index or type(call) is not int or not previous<call<=calls or event["result"]!=1
                or event["sample_end"]!=min(call*chunk,row["samples"]) or event["chunk_end"]!=call*chunk):
            raise ValueError("Invalid raw reference event identity or sample clock")
        previous=call
    if result["offsets"]!=[event["sample_end"] for event in events[:64]]:
        raise ValueError("Legacy prefix and full event stream differ")


def validate_ready(ready, engine, require_resources=False, require_threshold_control=False):
    if (ready.get("engine")!=engine or ready.get("mode")!="DET_MODE_95"
            or ready.get("rate")!=16000):
        raise RuntimeError("Reference firmware does not match the selected engine/mode/rate")
    if require_resources and ready.get("resource_probe") is not True:
        raise RuntimeError("Reference firmware is missing requested resource instrumentation")
    if ready.get("control_protocol")!=(CONTROL_PROTOCOL if require_threshold_control else None):
        raise RuntimeError("Reference threshold protocol and explicit runner mode differ")


def restoration_ready(value, expected_app, slot, service_id, require_online=True):
    return (value["app_sha256"]==expected_app and value["ota"]["running"]==slot
            and not value["ota"]["busy"] and not value["voice"]["active"]
            and not value["voice"].get("busy",False)
            and value["voice"]["wake"]["listening"]
            and value["voice"]["service_id"]==service_id
            and (not require_online or value["nyx"]["state"]=="up"))


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
    parser.add_argument("--engine", choices=("wn9_hiesp","wn9_heywillow_tts"), default="wn9_hiesp")
    parser.add_argument("--require-resources", action="store_true")
    parser.add_argument("--threshold-control", action="store_true",
                        help="Require official-threshold-v1 and explicit binary32 request on every manifest row")
    parser.add_argument("--restore-readiness", choices=("online","app-only"), default="online",
                        help="app-only explicitly excludes Nyx connectivity from an offline probe's restoration check")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baud", type=int, choices=(115200,921600,1500000,3000000), default=1500000)
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
    if before["voice"]["active"] or before["voice"].get("busy",False) or before["ota"]["busy"]:
        raise RuntimeError("Development board is busy")
    if bytes.fromhex(args.expected_app_sha256) not in args.restore_app.read_bytes():
        raise RuntimeError("Restore binary does not contain the expected ELF identity")
    if not all(0 < path.stat().st_size <= args.slot_bytes for path in (args.probe_bin,args.restore_app)):
        raise RuntimeError("Probe or restore app exceeds the explicitly verified slot size")
    rows = list(map(json.loads,args.manifest.read_text().splitlines()))
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        threshold_command(row,0,args.threshold_control)
        path = args.manifest.parent/row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Fixture WAV hash mismatch")
    chip = esptool(["chip_id"],"chip-private.log",30)
    if args.expected_mac.lower() not in chip.lower():
        raise RuntimeError("USB MAC does not match the development board")
    report = {"probe_sha256":hashlib.sha256(args.probe_bin.read_bytes()).hexdigest(),
              "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "engine":args.engine, "mode":"DET_MODE_95", "sample_rate":16000,
              "uart_baud":args.baud,
              "restore_readiness":args.restore_readiness,
              "before":{"app_sha256":before["app_sha256"],"slot":before["ota"]["running"],
                        "nyx_state":before["nyx"]["state"]},
              "scope":"Offline PCM detector benchmark, not acoustic/full-app acceptance.", "clips":[]}
    if args.threshold_control:
        report["control_protocol"]=CONTROL_PROTOCOL
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Benchmark interrupted; restoring original app")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        flash(args.probe_bin,"probe-flash-private.log")
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=30) as uart, (args.output/"uart-private.log").open("w") as log:
            uart.dtr=False;uart.rts=True;time.sleep(.1);uart.rts=False
            uart.reset_input_buffer()
            pending_events=[]
            def message(prefix, timeout=30):
                deadline=time.monotonic()+timeout
                while time.monotonic()<deadline:
                    line=uart.read_until(b"\n",8192).decode("utf-8",errors="replace").strip()
                    if line:
                        log.write(line+"\n");log.flush()
                    event_marker=line.find("WAKE_BENCH_EVENT ")
                    if event_marker>=0:
                        if not args.threshold_control:
                            raise RuntimeError("Unexpected controlled event stream in legacy mode")
                        pending_events.append(json.loads(line[event_marker+len("WAKE_BENCH_EVENT "):]))
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
            validate_ready(ready,args.engine,args.require_resources,args.threshold_control)
            if args.baud!=115200:
                uart.write(f"BAUD {args.baud}\n".encode("ascii"))
                assert message("WAKE_BENCH_BAUD")["baud"]==args.baud
                uart.baudrate=args.baud
                ready=message("WAKE_BENCH_READY",5)
                validate_ready(ready,args.engine,args.require_resources,args.threshold_control)
            report["firmware_ready"]=ready
            for index,row in enumerate(rows):
                with wave.open(str(args.manifest.parent/row["path"]),"rb") as wav:
                    if (wav.getframerate()!=16000 or wav.getnchannels()!=1 or wav.getsampwidth()!=2
                            or wav.getnframes()!=row["samples"] or wav.getcomptype()!="NONE"):
                        raise ValueError("Expected exact 16 kHz mono PCM16 fixture")
                    samples=wav.getnframes()
                    data=wav.readframes(samples)
                    if len(data)!=samples*2:
                        raise ValueError("Truncated WAV fixture")
                pending_events.clear()
                uart.write(threshold_command(row,index,args.threshold_control).encode("ascii"))
                prepared=message("WAKE_BENCH_CLIP_READY")
                assert prepared["id"] == index
                if args.threshold_control:
                    validate_control_initialization(prepared,row,index)
                for at in range(0,len(data),16384):
                    part=data[at:at+16384]
                    if uart.write(part)!=len(part):
                        raise RuntimeError("Short UART write")
                uart.flush()
                result=message("WAKE_BENCH_RESULT",max(30,samples/16000))
                if not (result["id"]==index and result["complete"] and result["crc32"]==zlib.crc32(data)):
                    raise RuntimeError("PCM transfer verification failed")
                clip={**row,"result":result,"threshold":prepared["threshold"]}
                if args.require_resources or args.threshold_control:
                    clip["initialization"]=prepared
                if args.threshold_control:
                    validate_control_result(result,pending_events,row,index,prepared["chunk"])
                    clip["raw_events"]=list(pending_events)
                report["clips"].append(clip)
                (args.output/"progress.json").write_text(json.dumps(report,indent=2)+"\n")
                if index%10==0 or index==len(rows)-1:
                    phase="offline_hiesp" if args.engine=="wn9_hiesp" else "offline_heywillow"
                    print(json.dumps({"phase":phase,"completed":index+1,"total":len(rows)}),flush=True)
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
                if restoration_ready(after,args.expected_app_sha256,args.slot,
                                     before["voice"]["service_id"],args.restore_readiness=="online"):
                    break
            except (OSError,ValueError):
                pass
            time.sleep(1)
        if not after or after["app_sha256"]!=args.expected_app_sha256 or after["ota"]["running"]!=args.slot:
            raise RuntimeError("RESTORE VERIFICATION FAILED")
        if not restoration_ready(after,args.expected_app_sha256,args.slot,
                                 before["voice"]["service_id"],args.restore_readiness=="online"):
            raise RuntimeError("Original app restored, but requested connection/service/idle wake state not ready")
        report["restored"]={"app_sha256":after["app_sha256"],"nyx_state":after["nyx"]["state"],
                            "slot":after["ota"]["running"],"voice_idle":True,
                            "service_selection_unchanged":True,"wake_listening":True,
                            "connection_verified":after["nyx"]["state"]=="up"}
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("Original app, service selection and Hi ESP listening verified; Nyx="+after["nyx"]["state"],flush=True)


if __name__ == "__main__":
    main()
