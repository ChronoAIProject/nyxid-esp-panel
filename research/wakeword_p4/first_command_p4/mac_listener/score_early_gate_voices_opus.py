"""Measure first-command retention at the new clocks and compare to the official Willow returns."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import torch

import check_early_gate_voices as voices
from score_paired_piper_willow import capture, host, uplink

ch, timing = voices.ch, voices.timing
OUT = ch.RESULTS / (voices.STEM + "-opus.json")


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    fw = uplink.firmware()
    reader, asr = ch.load_reader()
    decoder = uplink.decoder_api()
    previous = ch.read(voices.OUT)
    records, cases = [], []
    totals = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        for group, src, ref in (("paired",voices.study.matched.PLAN,voices.PAIRED),
                                ("connected",voices.connected.PLAN,voices.CONNECTED)):
            plan = ch.read(src)["records"]
            original = ch.read(ref)
            assert original["plan_sha256"]==ch.sha(src) and original["asr"]==asr and original["opus_sha256"]==fw["opus_sha256"]
            for r in previous["records"][group]:
                i = r["index"]
                s,old = plan[i],original["records"][i]
                assert i==old["index"]==s["index"] and s["pcm_sha256"]==r["pcm_sha256"]==old.get("mic_sha256",old.get("pcm_sha256"))
                pcm = host.piper.load_wav(s)
                assert ch.digest(pcm)==s["pcm_sha256"]
                modes = dict(r["routes"])
                official = old["routes"]["willow"]
                modes["willow"] = [timing.block(t) for t in official["events"]]
                role = s["target_id"]
                item = dict(group=group,index=i,voice=r["source"]["voice"],role=role,command=s["carrier"],routes={})
                openings = {}
                for mode,ts in modes.items():
                    total = totals[group][role][mode]
                    total["trials"] += 1
                    if not ts:
                        item["routes"][mode] = None
                        continue
                    opened = ts[0]
                    if opened not in openings:
                        cap_id = len(cases)
                        data = capture(reader,decoder,work,uart,cap_id,"voice",pcm,opened,s["carrier"])
                        data.update(id=cap_id,opened=opened,group=group,index=i,pcm_sha256=r["pcm_sha256"])
                        cases.append(data)
                        openings[opened] = data
                    c = openings[opened]
                    complete = c["lanes"]["selected"]["asr"]["complete_suffix"]
                    timely = s["start"]<=opened<s["end"]
                    if mode=="willow":
                        assert timely==official["timely"]
                        archived = official["capture"] if group=="paired" else old["captures"][str(opened)]
                        for name,lane in c["lanes"].items():
                            assert lane["decoded_sha256"]==archived["lanes"][name]["decoded_sha256"]
                    total.update(hits=1,timely=timely,complete=complete,timely_complete=timely and complete)
                    item["routes"][mode] = dict(opened=opened,lead=s["end"]-opened,capture=c["id"],complete=complete,timely=timely)
                records.append(item)
                if len(records)%40==0:
                    print(group,len(records),len(cases),flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for f in retries:
            uart.write(f,"retry/"+f.name)
    assert len(records)==222
    with ZipFile(archive.with_suffix(".tmp")) as uart:
        assert uart.testzip() is None
        for c in cases:
            index = c["id"]
            text = uart.read(f"opus/{index:03}-voice.log").decode().replace("\r\n","\n")
            for n,name in enumerate(("selected","raw")):
                packets = uplink.packets(text,index,c["emitted"]-c["first"],n)
                assert ch.digest(uplink.decode(decoder,packets))==c["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT,dict(records=records,captures=cases,summary=totals,asr=asr,archive_sha256=ch.sha(archive),
        opus_sha256=fw["opus_sha256"],qemu_sha256=fw["qemu_sha256"],retries=[f.name for f in retries],
        sources={p.name:ch.sha(p) for p in (voices.OUT,voices.PAIRED,voices.CONNECTED,Path(__file__))},
        policy="Both original synthetic corpora, two-second FIFO and existing silent-reference selected/raw audio. "
        "Encode each distinct per-source opening in P4 Opus. Re-decode both UART lanes. Count complete suffix "
        "from unprompted small.en on selected output. Original official Willow DEFAULT events on the identical "
        "source are block rounded and their decoded capture bytes must reproduce. Silent and cross-word controls included."))


if __name__=="__main__":
    main()
