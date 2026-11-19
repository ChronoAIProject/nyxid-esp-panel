"""Measure wall-clock OPEN and set-up delay with equal-priority P4 listen/encode tasks."""
from hashlib import sha256
from pathlib import Path
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import torch

import check_concurrent_speex_opus as concurrent

ch, uplink = concurrent.ch, concurrent.uplink
OUT = ch.RESULTS / "4090-c161-speex-opus-setup.json"


def main():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    parent = ch.read(concurrent.OUT)
    frozen = {(r["mic_sha256"],r["ref_sha256"]):r for r in parent["records"] if r["source"]["mode"]=="overlap"}
    assert ch.sha(concurrent.ARCHIVE) == parent["archive_sha256"]
    build = concurrent.BUILD
    app = build / "nyx_mac_listener.bin"
    work = uplink.listener.WORK / ("setup-"+ch.sha(app)[:16])
    work.mkdir(exist_ok=True)
    uplink.WORK = work
    model = (uplink.listener.WORK/"model/hey_nyx_model.bin").read_bytes()
    assert sha256(model).hexdigest()==parent["model_sha256"] and app.read_bytes().count(model)==1
    assert ch.sha(uplink.listener.QEMU) == parent["qemu_sha256"]
    decoder = uplink.decoder_api()
    archive = OUT.with_suffix(".zip")
    rows = []
    with ZipFile(concurrent.ARCHIVE) as old_logs, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as logs:
        for meta,mic,ref in concurrent.joint.fixed.spoken.extra_cases():
            if (meta.get("role"),meta["mode"],meta.get("external_to_echo_db")) != ("fixed-v5","overlap",0):
                continue
            key = ch.digest(mic),ch.digest(ref)
            old = frozen[key]
            assert old["source"] == meta
            i = old["index"]
            expected = concurrent.packets(old_logs.read(old["uart"]).decode().replace("\r\n","\n"))
            for ms,magic in ((0,0x4f52594e),(1500,0x5752594e)):
                path = work/f"{i:02}-{ms}-{key[0][:12]}.log"
                if not path.exists() or "LISTENER PASS" not in path.read_text():
                    data = struct.pack("<3I",magic,i,len(mic))+mic.tobytes()+ref.tobytes()
                    uplink.boot(build,path,b"LISTENER PASS",data,work)
                text = path.read_text()
                assert uplink.listener.values(text,"READY")==[i,len(mic),256,81600]
                opened,produced,first = uplink.listener.values(text,"OPEN")
                emitted,origin,count,blocks,total,peak,late,heap = uplink.listener.values(text,"FINISH")
                assert ([opened],first,first,count) == (old["open"],origin,old["first"],old["count"])
                assert count==emitted-first and produced-first==32000
                wall,launched,sent,requested = uplink.listener.values(text,"MIC_CLOCK")
                assert requested==ms*1000 and sent>=launched>=requested
                n,queued,caught,pending,backlog,worst,enc_total,stack,before,ready,left = uplink.listener.values(text,"MIC_QUEUE")
                packets = concurrent.packets(text)
                assert packets == expected and n==len(packets) and left==0 and caught>=sent
                digest = ch.digest(uplink.decode(decoder,packets))
                assert digest == old["decoded_sha256"]
                onset = meta["command_at"]*1000//16
                rows.append(dict(index=i,voice=meta["voice"],mic_sha256=key[0],ref_sha256=key[1],setup_ms=ms,
                                 open_sample=opened,first=first,emitted=emitted,command_at=meta["command_at"],
                                 wall_open_us=wall,wall_lead_us=onset-wall,worker_after_open_us=launched,
                                 first_packet_after_open_us=sent,first_packet_lead_us=onset-wall-sent,
                                 catch_up_after_open_us=caught,peak_fifo_samples=queued,max_pending_ticks=pending,
                                 ticks_with_backlog=backlog,blocks=blocks,blocks_over_16ms=late,peak_block_us=peak,
                                 mean_block_us=total/blocks,peak_encode_us=worst,total_encode_us=enc_total,
                                 min_internal=heap,stack_free=stack,internal_before_worker=before,internal_after_open=ready,
                                 packets=n,bytes=sum(map(len,packets)),decoded_sha256=digest,
                                 original_asr=old["asr"]["text"],complete=old["asr"]["complete_suffix"],
                                 uart=path.name,uart_sha256=ch.sha(path)))
                logs.write(path,path.name)
                print(i,ms,onset-wall,sent,caught,queued,flush=True)
        assert len(rows)==8
        for f in sorted(work.glob("*-attempt*.gz")):
            logs.write(f,f.name)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for r in rows:
            data = logs.read(r["uart"])
            assert sha256(data).hexdigest()==r["uart_sha256"]
            packets = concurrent.packets(data.decode().replace("\r\n","\n"))
            assert ch.digest(uplink.decode(decoder,packets))==r["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    folder = Path(__file__).parent
    ch.save(OUT,dict(records=rows,app_bytes=app.stat().st_size,app_sha256=ch.sha(app),
            model_sha256=parent["model_sha256"],qemu_sha256=parent["qemu_sha256"],asr=parent["asr"],
            images={f:ch.sha(build/f) for f in ch.read(build/"flasher_args.json")["flash_files"].values()},
            elf_sha256=ch.sha(build/"nyx_mac_listener.elf"),archive_sha256=ch.sha(archive),
            sources={p.name:ch.sha(p) for p in (Path(__file__),concurrent.OUT,folder/"main/capture_opus.h",folder/"main/listener.cpp")},
            retries=[p.name for p in sorted(work.glob("*-attempt*.gz"))],
            policy="Same four 0 dB fixed-cue double-talk inputs. Equal-priority 4 listening and Opus tasks. "
            "OPEN logged on the guest microsecond clock relative to starting the 16 ms periodic input timer. "
            "Compare immediately starting the encoder against deferring it until the first block at least "
            "1.5 seconds after OPEN. An active worker consumes the common FIFO and writes local packets. "
            "No real network or I2S. Match all packet bytes with the previous run, decode twice and attribute "
            "the existing ASR transcript only for the identical decoded PCM."))


if __name__=="__main__":
    main()
