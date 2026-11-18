"""Compare paired rhythm fits by the actual sentences gained and lost at fixed clocks."""
from collections import Counter
from pathlib import Path

from fit_c161_history_gain_v1 import RESULTS, read, save, sha

BASE = "4090-c161-midlevel"
PAIR = "4090-c161-rhythm-pair"
MODELS = {"midlevel":BASE,"native":PAIR+"-native","rhythm":PAIR+"-rhythm"}


def reports(cohort):
    result={}
    for name,stem in MODELS.items():
        path=RESULTS/f"{stem}-{cohort}-v1.json"
        data=read(path)
        archive=path.with_name(path.name.replace("-v1.json","-scores-v1.npz"))
        assert sha(archive)==data["scores_sha256"]
        result[name]=data["records"]
    return result


def main():
    summary={}
    commands=[]
    data=reports("context")
    for i,r in enumerate(data["midlevel"]):
        if "timeline" not in r:
            continue
        same={name:records[i] for name,records in data.items()}
        assert all(x["pcm_sha256"]==r["pcm_sha256"] and x["timeline"]==r["timeline"] for x in same.values())
        for span in r["timeline"]:
            if span["kind"]!="clip":
                continue
            arms={name:x["events"]["verifier"] for name,x in same.items()}
            arms["c161"]=r["events"]["c161"]
            in_span={name:{k:[t for t in ev if span["start"]<t<=span["end"]] for k,ev in e.items()}
                     for name,e in arms.items()}
            commands.append(dict(stream=r["stream"],**span,events=in_span))
    summary["context"]={}
    for group in ("nyx_connected_command","ordinary"):
        subset=[r for r in commands if (r["group"]=="nyx_connected_command")==(group=="nyx_connected_command")]
        summary["context"][group]={name:{k:sum(bool(r["events"][name][k]) for r in subset) for k in ("original","pair")}
                                   for name in ("c161",*MODELS)}
    for cohort in ("forward","transfer"):
        data=reports(cohort)
        support={}
        for name,rows in data.items():
            ref=data["midlevel"]
            assert [r["pcm_sha256"] for r in rows]==[r["pcm_sha256"] for r in ref]
            scored=[r for r in rows if r.get("target",r.get("role"))=="nyx"]
            support[name]={k:sum(any(r["start"]<t<=r["end"] for t in r[k]) for r in scored) for k in ("primary","pair")}
            support[name]["by_voice"]={v:sum(any(r["start"]<t<=r["end"] for t in r["primary"]) for r in scored if r["voice"]==v)
                                      for v in sorted({r["voice"] for r in scored})}
            if cohort=="transfer":
                support[name]["broadcasts"]={r["source_id"]:{k:r[k] for k in ("primary","pair")} for r in rows if r["group"]=="broadcast"}
        summary[cohort]=support
    for cohort in ("daily","mac","joe","willow","isolated-primed","isolated-jenny","isolated-melo"):
        data=reports(cohort)
        table={}
        for name,rows in data.items():
            assert [r["pcm_sha256"] for r in rows]==[r["pcm_sha256"] for r in data["midlevel"]]
            if cohort.startswith("isolated"):
                cnt=Counter((r["voice"].split(":")[-1],r["label"]) for r in rows if r["signed"])
                table[name]={v:{lbl:n for (voice,lbl),n in cnt.items() if voice==v} for v in sorted({v for v,_ in cnt})}
            elif cohort=="willow":
                table[name]={k:dict(target=sum(any(r["start"]<t<=r["end"] for t in r["events"]["verifier"][k]) for r in rows if r["role"]=="nyx"),
                                     pre=[r["key"] for r in rows if r["role"]=="nyx" and any(t<=r["start"] for t in r["events"]["verifier"][k])]) for k in ("original","pair")}
            else:
                table[name]={k:dict(Counter(r.get("target",r["text"]) for r in rows if r["events"]["verifier"][k]))
                             for k in ("original","pair")}
        summary[cohort]=table
    save(RESULTS/f"{PAIR}-review-v1.json",dict(summary=summary,connected=commands))


if __name__=="__main__":
    main()
