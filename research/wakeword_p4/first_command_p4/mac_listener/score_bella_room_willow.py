"""Score native Willow returns and send each actual opening through the product capture path."""
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_room_willow as setup
import check_bella_board_reference as board
import check_bella_reference_uplink as reference
from score_eight_voice_willow import timing

P, prior = reference.P, reference.prior
OUT = setup.PLAN.with_name("4090-c161-bella-room-willow.json")
BUILD = Path(".local/wakeword/c161-external-clock/build")


def official(source, mic_sha, plans, files, old_archives, archive, label):
    name, i = source["plan"], source["index"]
    path, plan = plans[name]
    record = plan["records"][i]
    assert record["pcm_sha256"] == mic_sha
    parts = {}
    for part in ("report.json","raw-returns.bin","serial.log"):
        if name == setup.PLAN.name:
            data = (setup.WORK/"runs"/f"{i:02}"/part).read_bytes()
        else:
            data = old_archives[name].read(f"{i:02}/willow-{part}")
        parts[part] = data
        setup.member(archive, f"{label}/{part}", data)
    r = json.loads(parts["report.json"])
    assert r["record_index"] == i and r["record"] == record and r["execution_sha256"] == files[name]
    assert r["model"] == "wn9_heywillow_tts" and r["mode"] == "DEFAULT" and r["stage"] == 104
    assert r["actual_float32_bits"] == r["post_prime_getter_bits"] == "3f24dd2f"
    for key in ("qemu","elf","flash"):
        assert r[key+"_sha256"] == plan["files"][plan["runtime"][key]]
    assert r["source_pcm_sha256"] == mic_sha and r["source_input_samples"] == record["samples"]
    assert r["total_prime_samples"] == 81600 and r["additional_prime_samples"] == 49344 and not r["transition_reset"]
    assert len(r["prime_returns"]) == 63 and all(v <= 0 for _,v in r["prime_returns"])
    assert not r["additional_priming_events"] and r["input_samples"] == record["samples"] + 49344
    for part, key in (("raw-returns.bin","raw_returns_sha256"),("serial.log","serial_sha256")):
        assert sha256(parts[part]).hexdigest() == r[key]
    raw = parts["raw-returns.bin"]
    assert len(raw) == r["input_calls"] == (r["input_samples"]+511)//512 and set(raw) <= {0,1,2}
    hits = [min(512*k,r["input_samples"])-49344 for k,v in enumerate(raw,1) if v == 2]
    assert hits == [e["relative_sample"] for e in r["real_events"]]
    opened = (hits[0]+255)//256*256 if hits else None
    return dict(native_events=hits, archive_prefix=label, **timing(record,opened))


def capture(work, archive, i, label, mic, ref, opened):
    assert 0 < opened <= ((len(mic)+255)//256)*256 and opened % 256 == 0
    payload = struct.pack("<3I",0x4b4c434e,i,len(mic))+mic.tobytes()+ref.tobytes()+struct.pack("<I",opened)
    def inspect(text):
        assert P.values(text,"READY") == [i,len(mic),256,81600]
        stop,first,count,blocks,*_ = P.values(text,"FINISH")
        assert P.values(text,"OPEN") == [opened,opened//160*160,first]
        assert blocks == (len(mic)+255)//256 and stop == blocks*256//160*160
        assert first == max(0,opened//160*160-32000) and count == stop-first
        audio = prior.samples(text,"AUDIO",count)
        assert np.array_equal(audio,prior.samples(text,"PROCESSED",count))
    text, uart = reference.verified_boot(work,archive,label+"/capture.log",BUILD,"nyx_mac_listener.bin",
                                         b"LISTENER PASS",payload,inspect)
    stop, first, count, *_ = P.values(text,"FINISH")
    pcm = prior.samples(text,"AUDIO",count)
    return pcm, dict(opened=opened,first=first,stop=stop,samples=count,fifo_sha256=P.digest(pcm),
                     capture_uart_sha256=uart,capture_watchdog=text.count("Task watchdog got triggered"))


def opus(work, archive, i, label, pcm, decoder):
    count = len(pcm)
    payload = struct.pack("<3I",0x3552504f,i,count) + pcm.tobytes()*2
    def inspect(text):
        assert prior.packets(text,i,count,0) == prior.packets(text,i,count,1)
    text, uart = reference.verified_boot(work,archive,label+"/opus.log",prior.CODEC,"nyx_first_command.bin",
                                         b"UPLINK PASS",payload,inspect)
    packets = prior.packets(text,i,count,0)
    audio = prior.decode(decoder,packets)
    return packets, audio, dict(opus_uart_sha256=uart,opus_watchdog=text.count("Task watchdog got triggered"),
                                packets=len(packets),bytes=sum(map(len,packets)),decoded_sha256=P.digest(audio))


def main():
    path, plan = setup.PLAN, P.read(setup.PLAN)
    known = P.read(reference.OUT)
    same_board = P.read(board.OUT)
    assert P.sha(reference.OUT) == plan["files"][str(reference.OUT)]
    assert P.sha(board.OUT) == plan["files"][str(board.OUT)]
    assert P.sha(prior.CODEC/"nyx_first_command.bin") == known["codec"]["opus_sha256"]
    assert P.sha(P.QEMU) == known["codec"]["qemu_sha256"]
    assert P.sha(reference.OUT.with_suffix(".zip")) == known["archive_sha256"]
    assert P.sha(setup.PACK) == plan["files"][str(setup.PACK)]
    plans, files, old_archives = {path.name:(path,plan)}, {path.name:P.sha(path)}, {}
    rows = []
    with ExitStack() as resources, TemporaryDirectory(dir=setup.WORK) as folder:
        work = Path(folder)
        for name in sorted({r[k]["plan"] for r in plan["cases"] for k in ("willow","no_address")} | {plan["quiet"]["source"]["plan"]}):
            if name == path.name:
                continue
            old = path.with_name(name)
            assert P.sha(old) == plan["files"][str(old)]
            results = old.with_name(old.name.replace("-plan.json",".json"))
            report = P.read(results)
            assert report["plan_sha256"] == P.sha(old) and report["archive_sha256"] == P.sha(results.with_suffix(".zip"))
            plans[name], files[name] = (old,P.read(old)), P.sha(old)
            old_archives[name] = resources.enter_context(ZipFile(results.with_suffix(".zip")))
        output = OUT.with_suffix(".zip")
        with ZipFile(work/output.name,"w",ZIP_DEFLATED) as raw:
            quiet = plan["quiet"]
            positive = official(quiet["source"],quiet["pcm_sha256"],plans,files,old_archives,raw,"quiet")
            assert positive["timely"]
            for i,(s,old) in enumerate(zip(plan["cases"],known["records"],strict=True)):
                wet = official(s["willow"],s["willow_mic_sha256"],plans,files,old_archives,raw,f"{i:02}/willow")
                erased = official(s["no_address"],s["no_address_mic_sha256"],plans,files,old_archives,raw,f"{i:02}/no-address")
                # The original control had a different address boundary but the same intact command clock.
                assert plans[s["willow"]["plan"]][1]["records"][s["willow"]["index"]]["end"] == s["end"]
                on, off = same_board["records"][2*i:2*i+2]
                assert on["label"] == "fixed/"+s["file"] and off["label"] == "no_address/"+s["file"]
                assert on["events"]["O"] == old["continuous"]["opened"] and old["timely"]
                assert (s["file"],s["end"],s["nyx_mic_sha256"],s["reference_sha256"]) == (
                    old["file"],old["command_at"],old["mic_sha256"],old["continuous"]["reference_sha256"])
                rows.append(dict(index=i,file=s["file"],db=s["db"],command=s["command"],command_at=s["end"],
                                 willow=wet,no_address=erased,nyx_open=on["events"]["O"],nyx_no_address=off["events"]["O"],
                                 nyx_qwen=old["product"]["qwen"]["complete"],nyx_turbo=old["product"]["turbo"]["complete_suffix"],
                                 product=None))
            opened = [r for r in rows if r["willow"]["opened"] is not None]
            identity = None
            if opened:
                decoder = prior.decoder_api()
                proc, model, qwen = prior.load()
                assert qwen == known["qwen"]
                turbo = prior.reader(known["turbo"])
                # Check the independent request clock on the first Willow hit's exact paired Nyx case.
                i = opened[0]["index"]
                old = known["records"][i]
                name = old["file"]
                _, mic_bytes, ref_bytes, _, _ = next(t for t in board.pairs() if t[0]=="fixed/"+name)
                mic, ref = (np.frombuffer(x,dtype="<i2") for x in (mic_bytes,ref_bytes))
                pcm, capture_id = capture(work,raw,i,"nyx-crosscheck",mic,ref,old["continuous"]["opened"][0])
                packets, _, codec_id = opus(work,raw,i,"nyx-crosscheck",pcm,decoder)
                assert capture_id["fifo_sha256"] == old["product"]["fifo_sha256"] and codec_id["decoded_sha256"] == old["product"]["decoded_sha256"]
                with ZipFile(reference.OUT.with_suffix(".zip")) as original:
                    data = original.read(f"{i:02}/aec.log")
                    assert sha256(data).hexdigest() == old["product"]["uart_sha256"]
                    assert packets == prior.packets(data.decode().replace("\r\n","\n"),i,len(pcm),0)
                identity = dict(index=i,file=name,**capture_id,**codec_id)
                panel, rate = sf.read(board.study.mixed.panel.demo.WAVE,dtype="int16")
                assert rate == 16000 and P.sha(board.study.mixed.panel.demo.WAVE) == known["panel_sha256"]
                for r in opened:
                    i = r["index"]
                    s = plan["cases"][i]
                    source = s["willow"]
                    wav = plans[source["plan"]][1]["records"][source["index"]]
                    mic, rate = sf.read(wav["path"],dtype="int16")
                    assert rate == 16000 and P.digest(mic) == s["willow_mic_sha256"]
                    ref = np.zeros_like(mic)
                    at = known["records"][i]["panel_at"]
                    ref[at:at+len(panel)] = panel
                    assert P.digest(ref) == s["reference_sha256"]
                    label = f"{i:02}/uplink"
                    pcm, capture_id = capture(work,raw,i,label,mic,ref,r["willow"]["opened"])
                    _, audio, codec_id = opus(work,raw,i,label,pcm,decoder)
                    r["product"] = dict(**capture_id,**codec_id,qwen=prior.evaluate(proc,model,audio,r["command"]),
                                        turbo=prior.recognition(turbo,audio,r["command"]))
        summary = dict(cases=len(rows),before=sum(r["willow"]["timely"] for r in rows),
                       after=sum(r["willow"]["opened"] is not None and r["willow"]["opened"] >= r["command_at"] for r in rows),
                       before_address=sum(r["willow"]["opened"] is not None and r["willow"]["opened"] < plan["cases"][r["index"]]["start"] for r in rows),
                       unopened=len(rows)-len(opened),no_address=sum(bool(r["no_address"]["native_events"]) for r in rows),
                       qwen=sum(r["product"]["qwen"]["complete"] for r in opened),
                       turbo=sum(r["product"]["turbo"]["complete_suffix"] for r in opened))
        assert summary["before"] + summary["after"] + summary["before_address"] + summary["unopened"] == len(rows) == 24
        assert len(opened) == sum(r["product"] is not None for r in rows)
        with ZipFile(work/output.name) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist())) and archive.testzip() is None
        P.save(work/OUT.name,dict(records=rows,willow_counts=summary,quiet=positive,nyx_identity_control=identity,
               plan_sha256=files[path.name],model=known["model"],nyx_source_sha256=P.sha(reference.OUT),
               board_source_sha256=P.sha(board.OUT),qwen=known["qwen"],turbo=known["turbo"],codec=known["codec"],
               capture_firmware_sha256=P.sha(BUILD/"nyx_mac_listener.bin"),archive_sha256=P.sha(work/output.name),
               sources={p.name:P.sha(p) for p in (Path(__file__),Path(__file__).with_name("main")/"listener.cpp",
                       Path(__file__).with_name("build_panel_marker_order.sh"))},
               policy="Classic ESP32 QEMU official DEFAULT on raw M. Count positive native returns after 81600 prime "
                      "samples. Round the first return up to the next completed 256-sample read. Use that request to "
                      "copy the two-second processed history in P4 QEMU and enable R for subsequent AEC reads. Feed "
                      "the whole FIFO through P4 Opus at 24 kb/s and decode the complete packets on the host. "
                      "Both GPU readers use the full decoded audio. Count misses in the 24-case denominator."))
        (work/output.name).replace(output)
        (work/OUT.name).replace(OUT)
        print(summary)


if __name__ == "__main__":
    main()
