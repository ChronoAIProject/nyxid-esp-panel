"""Verify candidate clocks and score ceilings; replay the complete matched sources."""
import argparse
import hashlib
from unittest.mock import patch

import numpy as np

from audit_c161_history_paths_v1 import arbitrate, decode, times
from check_c161_bounded_transfer_v1 import sources
from check_c161_speech_history_gate_v1 import history_gate
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_confuser_readout_v1 import STEM
from replay_c161_vowel_name_pair_v1 import binding
import verify_c161_ambient_stream_v1 as live


def transfer():
    prior = read(RESULTS/"4090-c161-bounded-readout-transfer-v1.json")
    archive = RESULTS/"4090-c161-bounded-readout-transfer-scores-v1.npz"
    assert sha(archive) == prior["scores_sha256"]
    native = binding()
    fit = read(RESULTS/f"{STEM}-training-v1.json")
    m, = fit["reports"]
    path = RESULTS/m["path"]
    assert sha(path) == m["sha256"]
    rows, traces = [], {}
    with np.load(archive) as ref:
        for (r,pcm), old in zip(sources(),prior["records"],strict=True):
            key = r["group"]+"_"+r["key"]
            assert old["pcm_sha256"] == r["pcm_sha256"] == hashlib.sha256(pcm.tobytes()).hexdigest()
            with patch.object(live, "MODEL", path):
                codes, clock, emitted, suppressed = live.run_stream(native, pcm)
            clock = np.asarray(clock)
            np.testing.assert_array_equal(ref[key+"_clock"],clock)
            scores = np.asarray(codes,np.uint8)
            assert np.all(scores <= ref[key+"_speech"])
            std, aux = [decode(v,clock) for v in scores]
            assert arbitrate(std,aux) == (emitted,suppressed)
            baseline = ref[key+"_c161"]
            gate,_ = history_gate(pcm,clock)
            selected = np.where(gate,scores[1],baseline[1])
            conditional = times(arbitrate(decode(baseline[0],clock),decode(selected,clock))[0])
            rows.append(dict(**r,primary=times(std),auxiliary=times(aux),pair=times(emitted),conditional=conditional))
            traces[key] = scores
    out = RESULTS/f"{STEM}-transfer-scores-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**traces)
    save(RESULTS/f"{STEM}-transfer-v1.json",dict(records=rows,scores_sha256=sha(out),model_sha256=sha(path)))


def audit():
    rows,summary = [],{}
    for cohort in ("context","daily","willow","mac","joe","generated"):
        prior = read(RESULTS/f"4090-c161-paired-event-readout-{cohort}-v1.json")
        current = read(RESULTS/f"{STEM}-{cohort}-v1.json")
        a = RESULTS/f"4090-c161-paired-event-readout-{cohort}-scores-v1.npz"
        b = RESULTS/f"{STEM}-{cohort}-scores-v1.npz"
        assert (sha(a),sha(b)) == (prior["scores_sha256"],current["scores_sha256"])
        steps = 0
        with np.load(a) as p, np.load(b) as z:
            for r,s in zip(current["records"],prior["records"],strict=True):
                k = r["key"]
                assert k == s["key"] and r["pcm_sha256"] == s["pcm_sha256"]
                np.testing.assert_array_equal(z[k+"_clock"],p[k+"_clock"])
                np.testing.assert_array_equal(z[k+"_c161"],p[k+"_c161"])
                assert np.all(z[k+"_verifier"] <= p[k+"_speech"])
                steps += len(z[k+"_clock"])
                if cohort == "context" and "timeline" in r:
                    for clip in r["timeline"]:
                        if clip["kind"] != "clip":
                            continue
                        paths = {}
                        for name,model in (("original",r["events"]["c161"]),("speech",s["events"]["speech"]),
                                           ("verifier",r["events"]["verifier"])):
                            paths[name] = {route:[t for t in ev if clip["start"] < t <= clip["end"]]
                                           for route,ev in model.items()}
                        rows.append(dict(stream=r["stream"], index=clip["text_index"],text=clip["text"],
                                         group=clip["group"],events=paths))
        summary[cohort] = dict(clocks=steps, streams=len(current["records"]), reference=sha(a),tested=sha(b))
    save(RESULTS/f"{STEM}-audit-v1.json",dict(clocks=summary,connected=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("transfer","audit"))
    {"transfer":transfer,"audit":audit}[parser.parse_args().action]()
