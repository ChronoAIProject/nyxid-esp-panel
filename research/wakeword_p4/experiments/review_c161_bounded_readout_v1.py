"""Preserve per-sentence results for the panel primary route and research dual route."""

import numpy as np

from audit_c161_history_paths_v1 import arbitrate, decode, times
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-bounded-readout"
PARENT = "4090-c161-paired-event-readout"


def main():
    ledger = []
    pins = {}
    for cohort in ("context", "daily", "willow", "mac", "joe", "generated"):
        report = read(RESULTS / f"{STEM}-{cohort}-v1.json")
        prior = read(RESULTS / f"{PARENT}-{cohort}-v1.json")
        archive = RESULTS / f"{STEM}-{cohort}-scores-v1.npz"
        reference = RESULTS / f"{PARENT}-{cohort}-scores-v1.npz"
        assert sha(archive) == report["scores_sha256"] and sha(reference) == prior["scores_sha256"]
        pins[cohort] = [sha(archive),sha(reference)]
        with np.load(archive) as z, np.load(reference) as p:
            for r, parent in zip(report["records"],prior["records"],strict=True):
                k = r["key"]
                assert k == parent["key"] and r["pcm_sha256"] == parent["pcm_sha256"]
                clock = z[k+"_clock"]
                np.testing.assert_array_equal(clock,p[k+"_clock"])
                np.testing.assert_array_equal(z[k+"_c161"],p[k+"_c161"])
                assert np.all(z[k+"_bounded"] <= p[k+"_speech"])
                ev = {}
                for name, scores, expected in (("c161",z[k+"_c161"],r),("speech",p[k+"_speech"],parent),
                                              ("free",z[k+"_free"],r),("bounded",z[k+"_bounded"],r)):
                    std, alt = [decode(c,clock) for c in scores]
                    ev[name] = dict(primary=times(std),pair=times(arbitrate(std,alt)[0]))
                    assert ev[name]["primary"] == expected["events"][name]["original"]
                    assert ev[name]["pair"] == expected["events"][name]["pair"]
                if cohort == "context" and "timeline" in r:
                    for clip in r["timeline"]:
                        if clip["kind"] != "clip":
                            continue
                        hits = {n:{m:[t for t in e if clip["start"]<t<=clip["end"]] for m,e in routes.items()}
                                for n,routes in ev.items()}
                        ledger.append(dict(cohort="connected",key=f'{r["stream"]}/{clip["text_index"]}',
                                           group=clip["group"],text=clip["text"],events=hits))
                else:
                    metadata = {key:r[key] for key in ("key","text","group","target","stream","role","start","end","voice","kind") if key in r}
                    ledger.append(dict(cohort=cohort,**metadata,events=ev))
    save(RESULTS/f"{STEM}-review-v1.json",dict(scores=pins,records=ledger))
    for route in ("primary","pair"):
        print(route)
        for cohort in ("connected","context"):
            rows = [r for r in ledger if r["cohort"]==cohort and r["group"]=="nyx_connected_command"]
            orig = {r["key"] for r in rows if r["events"]["c161"][route]}
            par = {r["key"] for r in rows if r["events"]["speech"][route]}
            for n in ("speech","free","bounded"):
                hits = {r["key"] for r in rows if r["events"][n][route]}
                print(cohort,n,len(hits),"gained",sorted(hits-orig),"lost",sorted(orig-hits),"vs parent",sorted(hits-par),sorted(par-hits))


if __name__ == "__main__":
    main()
