"""Score the frozen paired cues and pass captured audio through the verified P4 Opus firmware."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
import time
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import fixed_cue_dialogue_p4 as fixed
import paired_piper_willow as study
from score_handoff_willow import willow_events
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha

uplink, host = fixed.uplink, fixed.host


def capture(reader, decoder, work, archive, i, name, mic, opened, carrier, *, history_samples=32000):
    # Identical FIFO, 10 ms production and raw/90 ms delay with a silent reference.
    first = max(0, opened//160*160-history_samples)
    blocks = (len(mic)+255)//256
    emitted = blocks*256//160*160
    padded = np.pad(mic,(0,blocks*256-len(mic)))
    raw = padded[first:emitted]
    selected = np.pad(padded,(1440,0))[first:emitted]
    assert len(raw) == len(selected) and len(raw)>0
    payload = struct.pack("<3I",0x3552504f,i,len(raw))+selected.tobytes()+raw.tobytes()
    log = work/f"{i:03}-{name}.log"
    for attempt in range(3):
        uplink.boot(uplink.CODEC,log,b"UPLINK PASS",payload,work)
        text = log.read_text()
        try:
            frames = [uplink.packets(text,i,len(raw),lane) for lane in (0,1)]
            break
        except (AssertionError, ValueError):
            archive.write(log,f"retry/{log.name}-{attempt+1}")
            if attempt == 2:
                raise
    archive.write(log,"opus/"+log.name)
    log.unlink()
    lanes = {}
    for lane, signal in enumerate((selected,raw)):
        packets = frames[lane]
        audio = uplink.decode(decoder,packets)
        lanes[("selected","raw")[lane]] = dict(input_sha256=host.base.poll.digest(signal),
            decoded_sha256=host.base.poll.digest(audio),packets=len(packets),
            asr=uplink.recognition(reader,audio,carrier))
    return dict(first=first,emitted=emitted,lanes=lanes)


def main():
    fw = uplink.firmware()
    plan = read(study.PLAN)
    plan_hash = sha(study.PLAN)
    ref = read(RESULTS/(fixed.STEM+".json"))
    assert sha(MODEL)==MODEL_SHA
    assert fw["opus_sha256"]==ref["opus_sha256"]
    reader,identity = host.piper.load_reader()
    assert identity==ref["asr"]
    native,decoder = host.fixed.ch.c161_binding(),uplink.decoder_api()
    archive_path = RESULTS/(study.STEM+"-uart.zip")
    rows,summary,voices = [],defaultdict(Counter),defaultdict(Counter)
    with TemporaryDirectory() as tmp, ZipFile(archive_path.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        work = Path(tmp)
        uplink.WORK = work
        detector = host.fixed.ch.binding(work/"v5.so",coded="v5")
        for i,s in enumerate(plan["records"]):
            assert sha(Path(s["path"]))==s["sha256"] and i==s["index"]
            mic,rate = sf.read(s["path"],dtype="int16")
            assert rate==16000 and host.base.poll.digest(mic)==s["pcm_sha256"]
            # The official runner publishes one completed case directory at a time.
            report = study.WORK/"runs"/f"{i:02}"/"report.json"
            deadline = time.monotonic()+120
            while not report.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Missing completed Willow report {i}")
                time.sleep(.5)
            w = willow_events(i,s,plan_hash,z,runs=study.WORK/"runs")
            trace = host.fixed.ch.routes(detector,native,mic)
            assert trace["pcm_sha256"]==s["pcm_sha256"]
            routes,cache = {},{}
            for name,events in (("willow",w),("nyx",trace["merged"])):
                result = dict(events=events)
                if events:
                    opened = (events[0]+255)//256*256
                    lead = s["end"] - opened
                    if opened not in cache:
                        cache[opened] = capture(reader,decoder,work,z,i,name,mic,opened,s["carrier"])
                    result.update(open=opened,timely=s["start"]<=opened<s["end"],
                                  margin_samples=lead,capture=cache[opened])
                vals = dict(trials=1,hit=bool(events),timely=result.get("timely",False))
                vals["complete"] = bool(events and result["capture"]["lanes"]["selected"]["asr"]["complete_suffix"])
                vals["timely_complete"] = vals["timely"] and vals["complete"]
                for stats,k in ((summary,s["target_id"]),(voices,s["cue"])):
                    stats[k+"."+name].update(vals)
                routes[name] = result
            if s["cue"]=="fixed":
                previous = ref["records"][s["pair"]]
                p,old = routes["nyx"],previous["board"]
                assert trace["fixed"] and previous["mic_sha256"]==s["pcm_sha256"]
                assert p["open"]==old["delivered"] and p["capture"]["first"]==old["first"]
                for lane, data in p["capture"]["lanes"].items():
                    assert data["decoded_sha256"]==old["uplink"][lane]["decoded_sha256"]
                    assert data["input_sha256"]==old["capture_sha256"][lane]
                    assert data["asr"]["complete_suffix"]==old["uplink"][lane]["full"]["complete_suffix"]
            rows.append(dict(index=i,pair=s["pair"],cue=s["cue"],mic_sha256=s["pcm_sha256"],
                             command_sha256=s["suffix_sha256"],c161=trace["c161"],v5=trace["fixed"],routes=routes))
            if (i+1)%11==0:
                print(i+1,flush=True)
        failures=sorted(work.glob("*-attempt*.gz"))
        for f in failures:
            z.write(f,"failures/"+f.name)
    archive_path.with_suffix(".tmp").replace(archive_path)
    save(RESULTS/(study.STEM+".json"),dict(records=rows,summary=summary,voices=voices,asr=identity,
         plan_sha256=plan_hash,nyx_model_sha256=MODEL_SHA,qemu_sha256=fw["qemu_sha256"],
         opus_sha256=fw["opus_sha256"],uart_sha256=sha(archive_path),script_sha256=sha(Path(__file__)),
         codec_retry_logs=[p.name for p in failures],policy="Official ESP32 DEFAULT returns and independent frozen C161+v5 "
         "native events. Simulate common 256-sample delivery, 160-sample FIFO production and two seconds of "
         "pre-roll. Send both lanes to the real P4 QEMU Opus binary, decode and transcribe unprompted on GPU. "
         "For shipped v5, byte-validate every lane against the 18 earlier P4 listener+codec captures. A late "
         "notification may retain a complete command. No new hours or independent speakers from reusing commands."))


if __name__ == "__main__":
    main()
