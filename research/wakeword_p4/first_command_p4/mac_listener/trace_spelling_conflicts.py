"""Trace the old winning target path through each accepted daily or context window."""
from pathlib import Path

import numpy as np

import spelling_time_evidence as study
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

prior = study.prior


def trace(codes, word, vocab):
    costs = codes.astype(np.int32)
    costs -= costs.max(axis=1, keepdims=True)
    costs[:, 0] = np.maximum(costs[:, 0], costs[:, vocab["|"]])
    # Alternating blank and character states, as in the batched INT8 matcher.
    symbols = np.zeros(2 * len(word), dtype=int)
    symbols[1::2] = [vocab[ch] for ch in word]
    scores = np.full(len(symbols), -65536)
    parent = np.full((len(costs), len(symbols)), -1)
    totals = []
    for tick, frame in enumerate(costs):
        previous = scores.copy()
        for state, symbol in enumerate(symbols):
            choices = [(previous[state], state)]
            if state:
                choices.append((previous[state - 1], state - 1))
            if state >= 2 and symbol and symbol != symbols[state - 2]:
                choices.append((previous[state - 2], state - 2))
            if state == 1:
                choices.append((0, -1))
            value, parent[tick, state] = max(choices, key=lambda c: c[0])
            scores[state] = value + frame[symbol]
        totals.append(scores[-1])
    end = int(np.argmax(totals))
    tick, state = end, len(symbols) - 1
    states = []
    while state != -1:
        assert tick >= 0
        states.append((tick, state))
        state = parent[tick, state]
        tick -= 1
    assert sum(costs[t, symbols[s]] for t, s in states) == totals[end]
    chars = []
    for i, ch in enumerate(word):
        frames = [int(t) for t, s in states if s == 2 * i + 1]
        chars.append(dict(char=ch, first=min(frames), last=max(frames),
                          deficit=int(costs[frames, vocab[ch]].sum())))
    return dict(word=word, normalized=float(np.float32(totals[end] / len(word))),
                characters=chars, interword_blank_frames=chars[3]["first"] - chars[2]["last"] - 1,
                duration_frames=end - chars[0]["first"] + 1)


def main():
    vm, native = prior.model(), prior.fit.native_frontend()
    vocab = read(prior.WAV2VEC / "vocab.json")
    rows, inputs = [], {}
    for group in ("context", "daily"):
        files = [RESULTS / f"{stem}-{group}.json" for stem in (prior.STEM, prior.fit.STEM, study.STEM)]
        inputs.update({p.name: sha(p) for p in files})
        streams = zip(study.base.sources(group), *(read(p)["records"] for p in files), strict=True)
        for (_, pcm, _, _, _), old, letters, learned in streams:
            assert study.base.poll.digest(pcm) == old["pcm_sha256"] == learned["pcm_sha256"] == letters["pcm_sha256"]
            if not any(c["accepted"] for c in old["checks"]):
                continue
            raw, _, _ = study.base.native_inputs(native, pcm)
            for c, d, e in zip(old["checks"], letters["checks"], learned["checks"], strict=True):
                assert c["end"] == d["end"] == e["end"]
                if not c["accepted"]:
                    continue
                patch = np.rint(prior.fit.room.window(raw, c["end"]).clip(0, 666) * (255. / 666)).astype(np.uint8)
                codes = prior.outputs(vm, patch)
                student = d["models"]["student"]
                assert study.base.poll.digest(codes) == student["output_codes_sha256"]
                path = trace(codes, c["target"], vocab)
                assert path["normalized"] == c["target_value"]
                spans = [s for s in old["source"]["spans"] if s["start"] <= c["end"] < s["end"]]
                rows.append(dict(group=group, index=old["index"], stream=old["source"]["key"],
                                 pcm_sha256=old["pcm_sha256"], end=c["end"], spans=spans,
                                 target_path=path, teacher=d["teacher"], student=student, new_scores=e["scores"]))
    save(RESULTS / f"{study.STEM}-traces.json", dict(records=rows, inputs=inputs, script_sha256=sha(Path(__file__)),
         policy="Earliest maximal ending frame in each old accepted window. Trace one optimal alignment, preferring "
                "stay, advance by one, advance by two, then restart on ties. Frame indices zero-based, hop 20 ms. "
                "These are output frames with overlapping acoustic context, not phoneme boundaries."))


if __name__ == "__main__":
    main()
