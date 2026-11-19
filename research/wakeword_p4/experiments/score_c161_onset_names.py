"""Replay the frozen onset classifier, C161 and their union on original-speed streams."""
from pathlib import Path

import numpy as np
import torch

import check_c161_melo_address_grammar as melo
import check_c161_onset_names_melo as challenge
import check_c161_piper_spellings as piper
import check_c161_vad_pronunciations as vad
from check_c161_address_grammar import NAME_PLAN, merged, score_simple
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, read, save, sha
from review_c161_equal_length_history_v1 import replay
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_onset_names import CLASSES, Classifier, FIT, RESULTS, STEM, WEIGHTS, observations

PREVIOUS = RESULTS / "4090-c161-address-grammar.json"
PREVIOUS_MELO = RESULTS / f"{melo.STEM}.json"


def inputs():
    previous = {(r["cohort"], r["key"]):r for r in read(PREVIOUS)["records"]}
    piper_sources = read(piper.PLAN)["records"]
    for cohort, key, pcm, _, spans in vad.inputs():
        src = piper_sources[int(key)] if cohort == "piper" else dict(timeline=spans)
        yield cohort, key, pcm, src, previous[cohort,key]
    for cohort, plan, old in (("fresh", NAME_PLAN, previous), ("melo", melo.PLAN,
        {("melo",r["key"]):r for r in read(PREVIOUS_MELO)["records"]}), ("new_melo",challenge.PLAN,{})):
        for i,src in enumerate(read(plan)["records"]):
            key = str(i)
            yield cohort, key, load_wav(src), src, old.get((cohort,key))


def main():
    assert sha(MODEL) == MODEL_SHA and sha(WEIGHTS) == read(FIT)["weights_sha256"]
    assert read(challenge.PLAN)["archive_sha256"] == sha(challenge.ARCHIVE)
    torch.set_num_threads(1)
    model = Classifier().cuda().eval()
    with np.load(WEIGHTS) as data:
        model.load_state_dict({k:torch.from_numpy(data[k]) for k in data})
    reader, asr = load_reader()
    rows = []
    for cohort, key, pcm, src, prior in inputs():
        pcm_sha = digest(pcm)
        if prior is not None:
            assert prior["pcm_sha256"] == pcm_sha
        _, _, baseline = replay(pcm)
        if prior is not None:
            assert baseline == prior["events"]["c161"]
        candidates = observations(pcm)
        checks, accepted = [], []
        if candidates:
            # The first window in each stream must match a replay ending at that exact block.
            first_onset, first_end, first_patch = candidates[0]
            prefix = observations(pcm[:first_end])[0]
            assert prefix[:2] == (first_onset, first_end) and np.array_equal(first_patch,prefix[2])
            x = torch.from_numpy(np.stack([p for _,_,p in candidates])).cuda().float().unsqueeze(1)/255
            with torch.inference_mode():
                probs = model(x).softmax(dim=1).cpu().numpy()
            assert np.isfinite(probs).all()
            for (onset, end, patch), prob in zip(candidates, probs, strict=True):
                checks.append(dict(onset=onset, available=end, input_sha256=digest(patch),
                                   probabilities=dict(zip(CLASSES, prob.tolist(), strict=True))))
                if prob[0] >= .5:
                    accepted.append(end)
        events = dict(c161=baseline, names=merged([],accepted), combined=merged(baseline,accepted))
        if "carrier" in src:
            assert pcm_sha == src["pcm_sha256"]
            score = annotate(src,pcm,events,[],[],reader)
            for r in score["rows"]:
                del r["native_proposals"], r["expanded_proposals"]
        else:
            score = score_simple(src, events)
        rows.append(dict(cohort=cohort, key=key, pcm_sha256=pcm_sha, checks=checks, events=events, **score))
        if cohort != "daily":
            print(cohort,key,flush=True)
    refs = [FIT, WEIGHTS, PREVIOUS, PREVIOUS_MELO, piper.PLAN, NAME_PLAN, melo.PLAN, challenge.PLAN,
            Path(challenge.__file__), Path(__file__)]
    save(RESULTS/f"{STEM}-evaluation.json", dict(sources={p.name:sha(p) for p in refs}, c161_sha256=MODEL_SHA,
         asr=asr, records=rows, policy="One decision at each eligible onset + 21120 samples. Same 67-block merged cooldown. Two seconds of pre-roll for ASR. Full streams, host audio clock, no P4 scheduling."))


if __name__ == "__main__":
    main()
