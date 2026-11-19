"""Decode the original mixed-room product uplinks and test the receiver's existing two-second cut."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import check_bella_reference_uplink as reference
import score_bella_room_willow as comparison

P, prior = reference.P, reference.prior
OUT = P.RESULTS / "4090-c161-bella-product-history.json"


def captures():
    for label, file in (("nyx", reference.OUT), ("willow", comparison.OUT)):
        report = P.read(file)
        zipped = file.with_suffix(".zip")
        assert P.sha(zipped) == report["archive_sha256"]
        with ZipFile(zipped) as logs:
            for r in report["records"]:
                product = r["product"]
                if product is None:
                    yield dict(target=label,file=r["file"],opened=None,command=r["command"]), None
                    continue
                i = r["index"]
                if label == "nyx":
                    opened, = r["continuous"]["opened"]
                    first = r["continuous"]["finish"][1]
                    log = logs.read(f"{i:02}/aec.log")
                    uart = product["uart_sha256"]
                else:
                    opened = r["willow"]["opened"]
                    first = product["first"]
                    log = logs.read(f"{i:02}/uplink/opus.log")
                    uart = product["opus_uart_sha256"]
                assert sha256(log).hexdigest() == uart
                assert first == opened//160*160-32000 and opened < r["command_at"]
                packets = prior.packets(log.decode().replace("\r\n","\n"),i,product["samples"],0)
                assert len(packets) == product["packets"]
                row = dict(target=label,file=r["file"],db=r["db"],command=r["command"],opened=opened,
                           command_at=r["command_at"],first=first,packets=len(packets),
                           decoded_sha256=product["decoded_sha256"],full_qwen=product["qwen"]["complete"],
                           full_turbo=product["turbo"]["complete_suffix"])
                yield row, packets


def main():
    baseline = P.read(reference.OUT)
    processor, model, qid = prior.load()
    assert qid == baseline["qwen"]
    turbo = prior.reader(baseline["turbo"])
    decoder = prior.decoder_api()
    rows = []
    for row, packets in captures():
        if packets is not None:
            audio = prior.decode(decoder,packets)
            assert P.digest(audio) == row["decoded_sha256"]
            after_open = audio[32000:]
            reading = prior.evaluate(processor,model,after_open,row["command"])
            assert reading["pcm_sha256"] == P.digest(after_open)
            row.update(after_open_qwen=reading,after_open_turbo=prior.recognition(turbo,after_open,row["command"]))
        rows.append(row)
    counts = {}
    for target in ("nyx","willow"):
        opened = [r for r in rows if r["target"] == target and r["opened"] is not None]
        n = sum(r["target"] == target for r in rows)
        assert n == 24
        for reader, key in (("qwen","complete"),("turbo","complete_suffix")):
            pairs = [(r["full_"+reader],r["after_open_"+reader][key]) for r in opened]
            counts[target+"_"+reader] = dict(cases=n,opened=len(opened),full=sum(f for f,_ in pairs),
                                             after_open=sum(a for _,a in pairs),gained=pairs.count((False,True)),
                                             lost=pairs.count((True,False)))
    with TemporaryDirectory(dir=comparison.setup.WORK) as folder:
        output = Path(folder)/OUT.name
        P.save(output,dict(records=rows,counts=counts,qwen=qid,turbo=baseline["turbo"],
                           sources={p.name:P.sha(p) for p in (reference.OUT,comparison.OUT,Path(__file__))},
                           policy="Use each listener's actual request. Keep the entire 24 kb/s Opus stream and decode it "
                                  "from its beginning. Compare the original full ASR reading against the suffix from "
                                  "decoded sample index 32000. Reuse the previously fixed two-second cut for every open. "
                                  "End, readers and command references unchanged. Each denominator includes every non-open."))
        output.replace(OUT)
    print(counts)


if __name__ == "__main__":
    main()
