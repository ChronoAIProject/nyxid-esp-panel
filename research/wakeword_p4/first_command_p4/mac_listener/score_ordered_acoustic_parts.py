"""Replay both frozen integer acoustic-part models on consumed development speech and complete backgrounds."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import train_ordered_acoustic_parts as fit
from check_c161_event_transcripts_v1 import choose, event_inputs
from check_c161_room_word_background import native_features

RESULTS = fit.RESULTS
CONNECTED = RESULTS / "4090-c161-connected-nick-willow.json"
DAILY = RESULTS / "4090-c161-stream-stretch-daily-v1.json"
SIX = RESULTS / "4090-c161-six-cups-clock.json"


def corpus(mode, sources):
    def load(path):
        sources[path.name] = fit.sha(path)
        return fit.read(path)
    if mode == "speech":
        plan = CONNECTED.with_name(CONNECTED.stem+"-plan.json")
        report = load(CONNECTED)
        records = load(plan)["records"]
        assert fit.sha(plan) == report["plan_sha256"]
        for src,r in zip(records,report["records"],strict=True):
            assert src["pcm_sha256"] == r["pcm_sha256"]
            yield dict(group="connected",key=str(src["index"]),role=r["role"],text=src["synthesis"].get("text",r["carrier"]),
                       voice=r["voice"],start=src["start"],command_at=src["end"],willow=r["routes"]["willow"]["events"]), src,r["routes"]["c161"]["events"],None
        load(DAILY)
        for r,key,events,pcm in event_inputs("daily"):
            yield dict(group="daily",key=key,role="conversation",text=r["text"],voice=r["voice"]), r,events["c161"],pcm
        old = load(SIX)
        row, = [r for r in old["records"] if r["paired_alert_removed"]]
        wav = RESULTS / "c161-six-cups-false-wake.wav"
        assert fit.sha(wav) == old["wav_sha256"]
        yield dict(group="six_cups", key="whole",role="conversation",text=row.get("text","six cups")), dict(
            path=wav,pcm_sha256=row["pcm_sha256"]),row["events"]["M"],None
    for cohort in (("joe","jenny") if mode == "speech" else ("melo","primed","jenny")):
        prior = load(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")
        phase = "synthetic" if mode == "speech" else "background"
        for r in prior["records"]:
            if r["phase"] != phase:
                continue
            src = r["source"]
            yield dict(group=cohort, key=str(r["index"]),role=src.get("target_id","background"),
                       text=src.get("text"),voice=src.get("voice",cohort)),src,r["c161"],None


def events(scores, clock, cut):
    above = scores >= cut
    rises = np.flatnonzero(above & ~np.r_[False,above[:-1]])
    result = []
    for i in rises:
        if clock[i] > 0 and (not result or i-result[-1]>=67):
            result.append(int(i))
    return [int(clock[i]) for i in result]


def main(mode):
    fit_doc = fit.read(fit.FIT)
    native = fit.frontend.native_frontend()
    sources = {p.name:fit.sha(p) for p in (fit.FIT,Path(__file__))}
    models = {}
    for name,meta in fit_doc["models"].items():
        path = RESULTS / meta["file"]
        assert fit.sha(path) == meta["sha256"]
        with np.load(path, allow_pickle=False) as z:
            models[name] = (dict(z),meta)
    rows,summary = [],defaultdict(lambda:defaultdict(Counter))
    for row,source,c161,pcm in corpus(mode,sources):
        if pcm is None:
            path = Path(source.get("wav_path") or source["path"])
            pcm,rate = sf.read(path,dtype="int16")
            assert rate == 16000
        assert pcm.ndim == 1 and sha256(pcm.tobytes()).hexdigest() == source["pcm_sha256"]
        bands,clock = native_features(native,pcm)
        candidates = {}
        routes = {"c161":c161}
        for name,(weights,meta) in models.items():
            scores = fit.integer_scores(bands,weights,meta["gaps"])
            hits = events(scores,clock,meta["cutoff"])
            idx = int(np.argmax(np.where(clock>0,scores,-100000000)))
            candidates[name] = dict(score_sha256=sha256(scores.tobytes()).hexdigest(),
                                    peak=int(scores[idx]),peak_sample=int(clock[idx]),events=hits)
            routes[name] = hits
            routes["c161+"+name] = [v["sample"] for v in choose(c161,hits)]
        row.update(pcm_sha256=source["pcm_sha256"],samples=len(pcm),c161=c161,candidates=candidates,
                   union={k:v for k,v in routes.items() if k.startswith("c161+")})
        for name,ts in routes.items():
            ctr = summary[row["group"]+"."+row["role"]][name]
            ctr.update(cases=1,opened=int(bool(ts)),events=len(ts),samples=len(pcm),
                       gained=int(bool(ts) and not c161),lost=int(bool(c161) and not ts))
            if "command_at" in row:
                first = (ts[0]+255)//256*256 if ts else None
                ctr.update(timely=int(first is not None and row["start"]<=first<row["command_at"]))
        rows.append(row)
        if mode == "background" or len(rows)%100 == 0:
            print(row["group"],row["key"],len(rows),flush=True)
    expected = 60+96+1+360+396 if mode == "speech" else 66
    assert len(rows) == expected and len({(r["group"],r["key"]) for r in rows}) == expected
    if mode == "background":
        assert len({r["pcm_sha256"] for r in rows}) == 66
    output = RESULTS / (fit.STEM+"-"+mode+".json")
    with TemporaryDirectory(dir=fit.WORK) as folder:
        path = Path(folder)/output.name
        fit.save(path,dict(records=rows,summary=summary,sources=sources,models=fit_doc["models"],
                           c161=fit_doc["c161"],scope="Consumed development only. Complete file, one continuous prime. "
                           "Two-frame integer score, rising crossings and 67-call cooldown. Existing C161 events "
                           "on identical PCM. Timely uses the first 256-sample read ending strictly before command."))
        path.replace(output)


if __name__ == "__main__":
    mode, = sys.argv[1:]
    assert mode in ("speech","background")
    main(mode)
