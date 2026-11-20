"""Read both sender timing conditions at each actual native target opening."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import check_bella_weak_interference as baseline
import score_bella_weak_speaker_gain as scoring
from bella_address_gain import RESULTS, STEM, WORK

P = scoring.P
WAKE = RESULTS / (STEM+"-wake.json")
OUT = RESULTS / (STEM+"-commands.json")
TRACE = WORK / (STEM+"-sender-uart.zip")


def main():
    wake = P.read(WAKE)
    previous = P.read(baseline.OUT)
    known = {r["index"]:r for r in previous["records"] if r["condition"]=="speech1_music1"}
    assert P.sha(baseline.OUT.with_suffix(".zip")) == previous["archive_sha256"]
    artifacts = (Path(__file__), Path(scoring.__file__), WAKE, baseline.OUT, baseline.OUT.with_suffix(".zip"),
                 scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",scoring.sender.BUILD/"board.bin")
    sources = {p.name:P.sha(p) for p in artifacts}
    assert sources["nyx_mac_listener.bin"]==previous["capture_firmware_sha256"]
    assert sources["nyx_first_command.bin"]==previous["codec"]["opus_sha256"]
    assert sources["board.bin"]==previous["sender_firmware_sha256"]
    assert P.sha(P.QEMU)==previous["codec"]["qemu_sha256"]==wake["qemu_sha256"]
    processor, model, qwen = scoring.prior.load()
    assert qwen==previous["qwen"] and "4090" in qwen["device"]
    decoder = scoring.prior.decoder_api()
    decoder_version = decoder.opus_get_version_string().decode()
    assert decoder_version==P.read(scoring.OUT)["decoder"]
    readers = (processor,model,scoring.prior.reader(previous["turbo"]),decoder)
    cache = WORK/("sender-"+P.sha(WAKE)[:20])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows,summary = [],defaultdict(Counter)
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(baseline.OUT.with_suffix(".zip")) as historical:
        work = Path(tmp)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as logs:
            for i,r in enumerate(wake["records"]):
                role = r["role"]
                if role=="absent":
                    continue
                times = r["nyx"]["events"]["O"] if role=="nyx" else [t[1] for t in r["willow"]["events"]]
                opened = times[0] if times else None
                row = dict(label=r["label"],scene=r["scene"],role=role,level=r["level"],gain_db=r["gain_db"],
                           command=r["command"],end=r["end"],opened=opened,pcm_sha256=r["pcm_sha256"],
                           reference_sha256=r["reference_sha256"])
                c = summary[role+"/"+r["level"]]
                c.update(cases=1,opened=int(opened is not None),timely=int(opened is not None and r["start"]<=opened<r["end"]))
                if opened is not None:
                    earlier = known[r["scene"]]
                    row["reused"] = (r["pcm_sha256"],r["reference_sha256"],opened)==(
                                     earlier["mic_sha256"],earlier["reference_sha256"],earlier["opened"])
                    if row["reused"]:
                        reading = earlier
                        prefix = f"{r['scene']:03}/speech1_music1"
                        data = {s:historical.read(prefix+"/"+s+".log") for s in ("capture","opus","sender")}
                    else:
                        reading, shard = scoring.score_case(r,i,opened,readers,work,cache)
                        with ZipFile(shard) as fresh:
                            data = {s:fresh.read(r["label"]+"/"+s+".log") for s in ("capture","opus","sender")}
                    row["uart"] = {}
                    for stage,raw in data.items():
                        name = r["label"]+"/"+stage+".log"
                        digest = sha256(raw).hexdigest()
                        assert digest==reading[stage+"_uart_sha256"] and reading[stage+"_watchdog"]==0
                        logs.writestr(name,raw)
                        row["uart"][name]=digest
                    row.update(readings=reading["readings"],first=reading["first"],stop=reading["stop"],
                               samples=reading["samples"],fifo_sha256=reading["fifo_sha256"])
                    for label,asr in row["readings"].items():
                        c[label+"_qwen"] += int(asr["qwen"]["complete"])
                        c[label+"_turbo"] += int(asr["turbo"]["complete_suffix"])
                print(r["label"],opened,flush=True)
                rows.append(row)
        assert len(rows)==42 and all(c["cases"]==3 for c in summary.values())
        P.save(work/OUT.name,dict(records=rows,summary=summary,sources=sources,archive=TRACE.name,
               archive_sha256=P.sha(work/TRACE.name),qwen=qwen,turbo=previous["turbo"],decoder=decoder_version,
               scope="Three selected weak interruptions at seven levels per target. Actual first native opening "
                     "for each waveform. Full P4 capture, Opus and immediate/delayed sender. Original M/R and "
                     "unchanged dry command. Exact previously scored full-mixture endpoints reused."))
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
