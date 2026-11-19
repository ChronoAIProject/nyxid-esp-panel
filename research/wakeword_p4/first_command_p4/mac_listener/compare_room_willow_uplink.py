"""Compare published Willow decisions and both Nyx playback modes on matched room/command tracks."""
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

import check_v5_rnnoise_uplink as v5
import check_room_noise_frontends as ordinary
import score_room_open_context as context
from package_context_playback import add, formatted

ch, gap, room = ordinary.ch, ordinary.gap, ordinary.room
STEM = "4090-c161-fixed-spoken-willow-room-uplink"
OUT = ch.RESULTS/(STEM+".json")
WAV = ch.RESULTS/(STEM+"-pcm.zip")


def put_audio(pack, name, pcm):
    stream = io.BytesIO()
    sf.write(stream,pcm,16000,format="WAV",subtype="PCM_16")
    data = stream.getvalue()
    add(pack,name,data)
    return dict(file=name,samples=len(pcm),pcm_sha256=ch.digest(pcm),sha256=hashlib.sha256(data).hexdigest())


def main():
    rooms = room.rooms
    willow_file = ch.RESULTS/(rooms.STEM+"-willow.json")
    willow_uart = ch.RESULTS/(rooms.STEM+"-willow-uart.zip")
    plan = ch.read(rooms.PLAN)
    original = ch.read(willow_file)
    assert original["plan_sha256"] == ch.sha(rooms.PLAN) and original["uart_sha256"] == ch.sha(willow_uart)
    fixed = ch.read(room.parent.OUT)
    natural = ch.read(ordinary.OUT)
    readings = ch.read(context.OUT)
    info = readings["turbo"]
    reader, identity = ch.load_reader()
    assert identity == original["asr_files"] == readings["small"]
    assert all(ch.sha(context.prior.audit.TURBO/n)==h for n,h in info["files"].items())
    turbo = WhisperModel(str(context.prior.audit.TURBO),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder = gap.uplink.decoder_api()
    fw = gap.uplink.firmware()
    assert fw == natural["firmware"]
    srcs = list(ordinary.inputs())
    stats, rows, members = defaultdict(lambda:defaultdict(Counter)), [], []
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    wav_tmp = WAV.with_suffix(".tmp")
    nat_zip = ordinary.OUT.with_suffix(".zip")
    assert ch.sha(nat_zip) == natural["archive_sha256"]
    with TemporaryDirectory() as folder, ZipFile(willow_uart) as raw, ZipFile(nat_zip) as raw_nyx, \
            ZipFile(tmp,"w",ZIP_DEFLATED) as captures, ZipFile(wav_tmp,"w",ZIP_DEFLATED) as pack:
        work = Path(folder)
        gap.uplink.WORK = work
        for i, ((vs, vx), f) in enumerate(zip(v5.inputs(),fixed["records"],strict=True)):
            j = f["official_willow_source"]
            s, o = plan["records"][j], original["records"][j]
            folder_key = f"{j:02}/"
            report = json.loads(raw.read(folder_key+"report.json"))
            assert report["model"]=="wn9_heywillow_tts" and report["actual_float32_bits"]==report["post_prime_getter_bits"]=="3f24dd2f"
            assert report["stage"]==104 and report["total_prime_samples"]==81600 and report["execution_sha256"]==original["plan_sha256"]
            assert report["source_pcm_sha256"]==s["pcm_sha256"] and not report["additional_priming_events"]
            returns = raw.read(folder_key+"raw-returns.bin")
            assert hashlib.sha256(returns).hexdigest()==report["raw_returns_sha256"]
            events=(np.minimum((np.flatnonzero(np.frombuffer(returns,"u1")==2)+1)*512,report["input_samples"])-49344).tolist()
            assert events==o["events"]["willow"]==f["official_willow_events"]
            assert (s["voice"],s["carrier"],s["room"],s["end"])==(f["command_voice"],f["command"],f["room"],f["command_at"])
            assert ch.sha(Path(s["path"])) == s["sha256"]
            wx,rate=sf.read(s["path"],dtype="int16")
            assert rate==16000 and ch.digest(wx)==s["pcm_sha256"]
            (nx_meta,nx),(_,nick),(silent,z)=srcs[3*i:3*i+3]
            assert len(nx)==len(vx)==len(wx)==len(z) and ch.digest(z)==f["silence_sha256"]
            # Absolute times and the command/background construction come from the same pair.
            assert nx_meta["end"]==vs["end"]==s["end"]==silent["end"]
            assets={name:put_audio(pack,f"{i:02}/{name}.wav",x) for name,x in
                    (("fixed_v5",vx),("spoken_nyx",nx),("willow",wx),("nick",nick),("no_address",z))}
            members.append(dict(scene=i,room=s["room"],command=s["carrier"],command_voice=s["voice"],
                                command_at=s["end"],noise_sha256=f["noise_sha256"],rir_sha256=f["rir_sha256"],assets=assets))
            routes={}
            wopens=[context.prior.gap.study.voices.timing.block(t) for t in events if s["start"]<=t<len(wx)]
            nroute=natural["records"][3*i]["routes"]["raw"]
            fixed_read=readings["records"]["fixed"][str(vs["raw_capture"])]
            routes["fixed_v5"]=dict(opened=vs["opened"],full=fixed_read["full"],cropped=fixed_read["cropped"])
            for name,opens in (("willow",wopens),("spoken_nyx",[t for t in nroute["delivered"] if nx_meta["start"]<=t<len(nx)])):
                r = {}
                if opens:
                    at=opens[0]
                    if name=="willow":
                        cap=gap.replay.capture(reader,decoder,work,captures,j,"willow",wx,at,s["carrier"])
                        uart = captures.read(f"opus/{j:03}-willow.log")
                        r.update(capture=cap, id=j)
                        idx=j
                    else:
                        idx=nroute["capture"]
                        cap=natural["captures"][idx]
                        assert cap["input"]==3*i and cap["mode"]=="raw"
                        uart=raw_nyx.read(f"opus/{idx:03}-denoised.log")
                        r["prior_capture"]=idx
                    text=uart.decode().replace("\r\n","\n")
                    packets=gap.uplink.packets(text,idx,cap["emitted"]-cap["first"],0)
                    pcm=gap.uplink.decode(decoder,packets)
                    assert ch.digest(pcm)==cap["lanes"]["selected"]["decoded_sha256"]
                    cut=max(0,at-1600-cap["first"])
                    r.update(opened=at,decoded_sha256=ch.digest(pcm),full={},cropped={})
                    for view,signal in (("full",pcm),("cropped",pcm[cut:])):
                        for size,model in (("small",reader),("turbo",turbo)):
                            result = cap["lanes"]["selected"]["asr"] if (size,view)==("small","full") else gap.uplink.recognition(model,signal,s["carrier"])
                            r[view][size]={k:result[k] for k in ("text","complete_suffix","exact")}
                routes[name]=r
            rows.append(dict(scene=i,kind=vs["kind"],official_record=j,command=s["carrier"],voice=s["voice"],
                             command_at=s["end"],willow_events=events,willow_threshold=report["actual_threshold"],routes=routes))
            for name,r in routes.items():
                tally=stats[vs["kind"]][name]
                tally["trials"]+=1
                if "opened" in r:
                    tally.update(opens=1,timely=int(r["opened"]<s["end"]))
                    for view in ("full","cropped"):
                        for size in context.READERS:
                            tally[f"{size}_{view}_complete"]+=r[view][size]["complete_suffix"]
                            tally[f"{size}_{view}_timely_complete"]+=r[view][size]["complete_suffix"] and r["opened"]<s["end"]
        add(pack,"manifest.json",formatted(dict(rate=16000,format="mono PCM16LE WAV",records=members,
            purpose="Digital microphone injection. Files already include the virtual room and noise. "
            "They pair the same command, voice, timing and background across all five addresses/controls.")))
        for p in sorted(work.glob("*-attempt*.gz")):
            captures.write(p,"retry/"+p.name)
    assert len(rows)==16
    with ZipFile(wav_tmp) as pack:
        assert pack.testzip() is None
        for m in json.loads(pack.read("manifest.json"))["records"]:
            for item in m["assets"].values():
                data=pack.read(item["file"])
                assert hashlib.sha256(data).hexdigest()==item["sha256"]
                x,sr=sf.read(io.BytesIO(data),dtype="int16")
                assert sr==16000 and ch.digest(x)==item["pcm_sha256"]
    with ZipFile(tmp) as logs:
        assert logs.testzip() is None
        for row in rows:
            r=row["routes"]["willow"]
            if "capture" in r:
                j,cap=r["id"],r["capture"]
                data=logs.read(f"opus/{j:03}-willow.log").decode().replace("\r\n","\n")
                for lane,name in enumerate(("selected","raw")):
                    packets=gap.uplink.packets(data,j,cap["emitted"]-cap["first"],lane)
                    assert ch.digest(gap.uplink.decode(decoder,packets))==cap["lanes"][name]["decoded_sha256"]
    wav_tmp.replace(WAV)
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows, summary=stats,archive_sha256=ch.sha(archive),audio_sha256=ch.sha(WAV),
        sources={p.name:ch.sha(p) for p in (rooms.PLAN,willow_file,willow_uart,room.parent.OUT,context.OUT,ordinary.OUT,nat_zip,Path(__file__))},
        firmware=fw,turbo=info,small=identity,policy="Official Willow event clock from published emulated ESP32 WakeNet returns. "
        "Same P4 codec, two-second pre-roll, selected lane, unprompted ASR and score. Each system receives its own cue. "
        "Both Nyx cues are repeated across commands, Willow is generated in each of four paired command voices. "
        "Microphone-level digital comparison, no independent hardware or human generalization."))


if __name__ == "__main__":
    main()
