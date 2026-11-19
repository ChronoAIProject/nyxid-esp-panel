"""Attribute the first in-span notification in every published route and source span."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

RESULTS=Path("research/wakeword_p4/results")
STEM="4090-c161-polling-rooms"
GROUPS=("fresh","context","daily","bot")
TARGETS=("Hey Nyx","Hye Nyx","Hey Nicks","Hi Nicks")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    joint=RESULTS/(STEM+"-proposals-agreement.json")
    joined=read(joint)
    sources={joint.name:sha(joint)}
    details=[]
    totals=defaultdict(lambda:dict(inputs=0,hits=0,c161_first=0,auxiliary_first=0,exact_command=0,asr_scored=0))
    for group in GROUPS:
        original=RESULTS/(STEM+"-"+group+".json")
        conditional=RESULTS/(STEM+"-proposals-"+group+".json")
        report=read(conditional)
        assert sha(original)==report["prior_sha256"]
        assert sha(conditional)==joined["source_reports"][group]
        sources.update({original.name:sha(original),conditional.name:sha(conditional)})
        for base,checks,strict in zip(read(original)["records"],report["records"],joined["records"][group],strict=True):
            assert base["index"]==checks["index"]==strict["index"] and base["pcm_sha256"]==checks["pcm_sha256"]==strict["pcm_sha256"]
            stream=base["routes"]|checks["routes"]|strict["routes"]
            for a,b,c in zip(base["rows"],checks["rows"],strict["rows"],strict=True):
                assert len({(v["address"],v["start"],v["end"]) for v in (a,b,c)})==1
                address=a["address"]
                terms=a["routes"]|b["routes"]|c["routes"]
                items={}
                for route,r in terms.items():
                    first=min(r["hits"]) if r["hits"] else None
                    origin=None if first is None else ("C161" if first in stream["c161"] else "auxiliary")
                    if first is not None:
                        assert first in stream[route]
                    items[route]=dict(first=first,origin=origin,exact_command=r.get("exact_command"))
                    acoustic=base["meta"].get("acoustic","original")
                    labels=[address]
                    if group=="fresh":
                        if address in TARGETS:
                            labels.append("all_Nyx_spellings")
                        elif address not in ("Hey Willow","daily"):
                            labels.append("all_competing_addresses")
                    if group=="context" and address.startswith("ordinary_"):
                        labels.append("all_ordinary_context")
                    for label in labels:
                        for voice in (["all",base["meta"]["voice"]] if group=="fresh" else ["all"]):
                            stat=totals[group,acoustic,voice,label,route]
                            stat["inputs"]+=1
                            stat["hits"]+=first is not None
                            stat["c161_first"]+=origin=="C161"
                            stat["auxiliary_first"]+=origin=="auxiliary"
                            stat["asr_scored"]+="exact_command" in r
                            stat["exact_command"]+=r.get("exact_command",False)
                details.append(dict(group=group,record=base["index"],meta=base["meta"],address=address,
                    start=a["start"],end=a["end"],pcm_sha256=base["pcm_sha256"],routes=items))
    counts=[]
    for (group,acoustic,voice,label,route),stat in sorted(totals.items()):
        assert stat["hits"]==stat["c161_first"]+stat["auxiliary_first"]
        counts.append(dict(group=group,acoustic=acoustic,voice=voice,label=label,route=route,**stat))
    return dict(sources=sources,records=details,counts=counts,
        policy="First emitted notification inside each annotated span, original C161 wins ties on the same sample. Complete command is the previously scored two-second pre-roll transcript suffix where ASR exists. All rows include stream PCM identity.")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    destination=parser.parse_args().output
    with destination.open("x") as handle:
        json.dump(summarize(),handle,indent=2)
        handle.write("\n")
