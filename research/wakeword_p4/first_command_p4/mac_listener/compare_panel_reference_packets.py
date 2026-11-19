"""Check the complete paired Opus payloads on common OPENs across reference modes."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import check_panel_middle_p4 as middle

ch, tx = middle.ch, middle.tx
OUT=ch.RESULTS/"4090-c161-panel-reference-packet-comparison.json"


def main():
    baseline=ch.read(tx.diag.BOARD)
    base={r["index"]:r for r in baseline["records"] if r["name"]=="bca" and r["role"]=="both"}
    intact=ch.read(tx.balanced.OUT)
    physical={base[r["baseline_index"]]["host_index"]:r for r in intact["records"] if r["mode"]=="balanced"}
    missing=ch.read(middle.pair.BOARD)
    paired={r["host_index"]:r for r in missing["records"] if r["group"].startswith("balanced.")}
    curr=ch.read(middle.OUT)
    summary=defaultdict(Counter)
    pairs=[]
    docs=(tx.balanced.OUT,middle.pair.BOARD,middle.OUT)
    for doc,parsed in zip(docs,(intact,missing,curr),strict=True):
        assert ch.sha(doc.with_suffix(".zip"))==parsed["archive_sha256"]
    with ZipFile(docs[0].with_suffix(".zip")) as a, ZipFile(docs[1].with_suffix(".zip")) as b, ZipFile(docs[2].with_suffix(".zip")) as c:
        for r in curr["records"]:
            group=r["group"].split(".")[0]
            if group not in ("physical","balanced"): continue
            i=r["host_index"]
            prior,archive=(physical[i],a) if group=="physical" else (paired[i],b)
            assert all(r[k]==prior[k] for k in ("source","mic_sha256","reference_sha256"))
            same=r["opened"]==prior["opened"]
            equal=0
            if same and r["opened"]:
                def packets(z,row):
                    uart=z.read(f'{row["index"]:02}-opus.log').decode().replace("\r\n","\n")
                    return tx.panel.grid.gap.uplink.packets(uart,row["index"],row["uplink"]["samples"],0)
                x,y=packets(c,r),packets(archive,prior)
                assert x==y and r["uplink"]["pcm_sha256"]==prior["uplink"]["pcm_sha256"]
                assert (r["small"],r["turbo"])==(prior["small"],prior["turbo"])
                equal=len(x)
            pairs.append(dict(group=group,host_index=i,previous_index=prior["index"],index=r["index"],
                              prior_open=prior["opened"],open=r["opened"],equal_packets=equal))
            summary[group].update(trials=1,same_open=int(same),equal_packets=equal)
    assert len(pairs)==72
    timing={}
    for k,report in (("timer",ch.read(tx.BOARD)),("pair",missing),("middle",curr)):
        runs=report["records"]
        timing[k]=dict(cases=len(runs),blocks=sum(r["finish"][3] for r in runs),
                       peak_us=max(r["finish"][5] for r in runs),over_16ms=sum(r["finish"][6] for r in runs),
                       min_internal_heap=min(r["finish"][7] for r in runs))
    ch.save(OUT,dict(records=pairs,summary=summary,timing=timing,
        sources={p.name:ch.sha(p) for p in (*docs,tx.BOARD,Path(__file__))},
        policy="Compare 60 missing-R balanced command streams against the acoustic pair receiver and 12 intact-R "
               "simultaneous streams against the tested original electrical receiver. Check the complete selected "
               "Opus bytes, not just counts, when both devices OPEN at the same time."))


if __name__=="__main__":
    main()
