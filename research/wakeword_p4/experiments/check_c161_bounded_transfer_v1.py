"""Replay the two complete broadcast conflicts and all 24 four-voice streams."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit_c161_history_paths_v1 import arbitrate, decode, times
from check_c161_speech_history_gate_v1 import history_gate
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-bounded-readout"


def sources():
    broadcasts = read(RESULTS / "4090-c161-history-gain-melo-v1.json")["records"]
    conflicts = read(RESULTS / "4090-c161-dual-readout-background-melo-v1.json")["records"]
    targeted = {r["index"]: r for r in conflicts if r["events"]["always"]}
    assert set(targeted) == {458, 471}
    for ref in broadcasts:
        if ref["phase"] == "background" and ref["index"] in targeted:
            s = ref["source"]
            track = targeted[ref["index"]]
            assert s["pcm_sha256"] == track["pcm_sha256"]
            yield dict(key=str(ref["index"]), group="broadcast", source_id=track["source_id"],
                       history_events=track["events"], pcm_sha256=s["pcm_sha256"]), load_wav(s)
    plan = read(RESULTS / "4090-c161-four-voice-forward-v1.json")
    for i, r in enumerate(plan["records"]):
        yield dict(key=str(i), group="forward", voice=r["voice"], scene=r["scene"], target=r["target_id"],
                   start=r["start"], end=r["end"], first_phone=r["first_phone"], official=r["official_events"],
                   pcm_sha256=r["pcm_sha256"]), load_wav(r)


def main():
    fit = read(RESULTS / f"{STEM}-training-v1.json")
    assert sha(MODEL) == fit["baseline_sha256"] == MODEL_SHA
    parent = RESULTS / "4090-c161-paired-event-readout-speech-v1.tflite"
    assert sha(parent) == fit["parent_sha256"]
    paths = dict(c161=MODEL, speech=parent)
    for r in fit["reports"]:
        path = RESULTS / r["path"]
        assert sha(path) == r["sha256"] and path.stat().st_size == 60952
        paths[r["name"]] = path
    native = binding()
    rows, traces = [], {}
    for r, pcm in sources():
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
        key = r["group"] + "_" + r["key"]
        scores = {}
        events = {}
        for name, path in paths.items():
            with patch.object(live, "MODEL", path):
                output, clock, emitted, suppressed = live.run_stream(native,pcm)
            if name == "c161":
                traces[key+"_clock"] = np.asarray(clock)
            np.testing.assert_array_equal(clock, traces[key+"_clock"])
            scores[name] = np.asarray(output,np.uint8)
            standard, auxiliary = [decode(x,traces[key+"_clock"]) for x in scores[name]]
            assert arbitrate(standard,auxiliary) == (emitted,suppressed)
            events[name] = dict(primary=times(standard), auxiliary=times(auxiliary),pair=times(emitted))
            traces[key+"_"+name] = scores[name]
        assert np.all(scores["bounded"] <= scores["speech"])
        clock = traces[key+"_clock"]
        gate, _ = history_gate(pcm,clock)
        for name in ("speech","free","bounded"):
            selected = np.where(gate,scores[name][1],scores["c161"][1])
            first = decode(scores["c161"][0],clock)
            events[name]["conditional"] = times(arbitrate(first,decode(selected,clock))[0])
        if r["group"] == "broadcast":
            assert events["speech"]["conditional"] == r["history_events"]["conditional"]
        rows.append(dict(**r,events=events))
        print(key,flush=True)
    file = RESULTS/f"{STEM}-transfer-scores-v1.npz"
    with file.open("xb") as f:
        np.savez_compressed(f,**traces)
    save(RESULTS / f"{STEM}-transfer-v1.json",dict(scores_sha256=sha(file), models={k:sha(p) for k,p in paths.items()},records=rows))


if __name__=="__main__":
    main()
