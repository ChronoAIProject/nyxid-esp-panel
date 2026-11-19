"""Pair each latent-alignment trial before and after refreshing the training negatives."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2]/"results"
STEM = "4090-c161-connected-alignment"


def compare():
    inputs = {}
    def load(suffix):
        path = RESULTS/(STEM+suffix+".json")
        raw = path.read_bytes()
        inputs[path.name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    replays = [load("-replay"),load("-refreshed-replay")]
    commands = [load("-commands"),load("-refreshed-commands")]
    backgrounds = [load("-background"),load("-refreshed-background")]
    arms = ("latent_confirmed","refreshed_confirmed")
    groups,rows = defaultdict(Counter),[]
    def tally(key, before, after):
        groups[key]["both" if before and after else "lost" if before else "gained" if after else "neither"] += 1
    command_rows = [{r["key"]:r for r in d["records"]} for d in commands]
    for left,right in zip(*(d["records"] for d in replays),strict=True):
        assert all(left.get(k)==right.get(k) for k in ("key","samples","source","cohort","group","text","pcm_sha256"))
        pair = (left,right)
        info = {k:left[k] for k in ("key","group","cohort","track","source","text","pcm_sha256") if k in left}
        if "track" in left:
            streams = [r["opened"][a] for r,a in zip(pair,arms)]
            info["opened"] = streams
            if left["track"]=="plain":
                scores = []
                for j,(r,a,opened) in enumerate(zip(pair,arms,streams,strict=True)):
                    s = r["source"]
                    timely = bool(opened and s["start"]<=opened[0]<s["command_at"])
                    case = command_rows[j].pop(r["key"])["captures"][a]
                    assert (case is None) == (not opened)
                    c = commands[j]["captures"][case] if case is not None else None
                    if c is not None:
                        assert (c["mic_sha256"],c["opened"])==(r["pcm_sha256"],opened[0])
                    scores.append(dict(timely=timely,timely_qwen=bool(timely and c["qwen"]["complete"]),
                                       timely_turbo=bool(timely and c["turbo"]["complete_suffix"]),capture=case))
                for field in ("timely","timely_qwen","timely_turbo"):
                    tally(field,scores[0][field],scores[1][field])
                info["first_commands"] = scores
            else:
                tally("no_address",*map(bool,streams))
        else:
            streams = [r["events"][a] for r,a in zip(pair,arms)]
            info["events"] = streams
            if left["group"]=="common":
                tally("common."+left["text"],*map(bool,streams))
            else:
                for span in left["source"]["spans"]:
                    flags = [any(span["start"]<=t<span["end"]+6400 for t in hits) for hits in streams]
                    tally(span["address"],*flags)
        rows.append(info)
    assert all(not index for index in command_rows)
    for l,r in zip(*(d["records"] for d in backgrounds),strict=True):
        assert (l["key"],l["pcm_sha256"],l["samples"])==(r["key"],r["pcm_sha256"],r["samples"])
        hits = [d["events"][a] for d,a in zip((l,r),arms)]
        tally("background_file",*map(bool,hits))
        rows.append(dict(group="background",key=l["key"],pcm_sha256=l["pcm_sha256"],samples=l["samples"],events=hits))
    return dict(arms=arms, sources=inputs,summary=groups, records=rows)


if __name__=="__main__":
    path = RESULTS/(STEM+"-paired-refresh.json")
    doc = compare()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc,indent=2)+"\n")
    tmp.replace(path)
