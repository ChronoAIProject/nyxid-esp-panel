"""Read two causal proposal windows with the fixed English GPU CTC teacher."""
from pathlib import Path

import numpy as np
import torch
from tempfile import TemporaryDirectory

import music_proposal_replay as base
import proposal_lexical_readout as lexical
import train_positive_readout as fit
from check_c161_event_transcripts_v1 import PHRASE
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from score_proposal_readout import GROUPS

STEM = "4090-c161-proposal-ctc"
WIDTH = 144*160


@torch.inference_mode()
def decode(model, proc, clip):
    sample = proc(clip.astype(np.float32)/32768, sampling_rate=16000, return_tensors="pt").input_values.cuda()
    ids = model(sample).logits[0].argmax(dim=1).cpu()
    text = proc.decode(ids)
    return dict(text=text, accepted=bool(PHRASE.search(text)), pcm_sha256=base.poll.digest(clip))


def main(out=RESULTS):
    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu and sha(MODEL) == MODEL_SHA
    model, proc, model_info = fit.parent.room.BASE.teacher()
    for group in GROUPS:
        word_file = RESULTS/f"{lexical.STEM}-{group}.json"
        net_file = RESULTS/f"{fit.STEM}-{group}.json"
        records = []
        for (_, pcm, c161, _, _), w, n in zip(base.sources(group),read(word_file)["records"],read(net_file)["records"],strict=True):
            assert base.poll.digest(pcm) == w["pcm_sha256"] == n["pcm_sha256"] and c161 == n["c161"] == w["c161"]
            by_proposal = {c["proposal"]:c for c in n["checks"]}
            checks = []
            accepted = {"neural_window":[], "lexical_window":[], "ctc_or_joint":[]}
            for c in w["checks"]["name-contrast"]:
                t, lo, end = c["proposal"], c["start"], c["end"]
                row = dict(proposal=t, lexical_start=lo, lexical_end=end, q8=c["confirmed"])
                row["lexical_window"] = decode(model, proc, pcm[lo:end])
                if row["lexical_window"]["accepted"]:
                    accepted["lexical_window"].append(end)
                if t in by_proposal:
                    at = by_proposal[t]["end"]
                    begin = at-WIDTH
                    clip = pcm[max(0,begin):at]
                    if begin < 0:
                        clip = np.pad(clip,(-begin,0))
                    assert len(clip) == WIDTH
                    row.update(verifier_end=at, verifier_start=begin, verifier_codes=by_proposal[t]["codes"]["trigger"],
                               neural_window=decode(model,proc,clip))
                    if row["neural_window"]["accepted"]:
                        accepted["neural_window"].append(at)
                        accepted["ctc_or_joint"].append(at)
                    if c["confirmed"] is not None and row["verifier_codes"][0] >= 0:
                        accepted["ctc_or_joint"].append(max(c["confirmed"], at))
                checks.append(row)
            routes = {k:base.scoring.merged(c161,events) for k,events in accepted.items()}
            r = dict(index=w["index"], pcm_sha256=w["pcm_sha256"], samples=len(pcm), c161=c161,
                     checks=checks, routes=routes, source=w["source"])
            if group == "context":
                r["full_addresses"] = [dict(start=s["start"],end=s["end"],text=s["text"],
                                       ctc=decode(model,proc,pcm[s["start"]:s["end"]]))
                                       for s in w["source"]["spans"] if s["text"].startswith("Hey ")]
            elif group == "background":
                r["key"] = w["key"]
            elif group == "fresh":
                r["spans"] = [{k:s[k] for k in ("address","start","end","pcm_sha256")} for s in n["rows"]]
            records.append(r)
        save(out/f"{STEM}-{group}.json",dict(records=records, gpu=gpu, torch=torch.__version__, teacher=model_info,
             pattern=PHRASE.pattern, script_sha256=sha(Path(__file__)), c161_sha256=MODEL_SHA,
             sources={p.name:sha(p) for p in (word_file,net_file)},
             policy="Frozen English wav2vec2-base-960h, greedy transcript with the existing wake regex. Exactly "
             "the lexical search PCM or a 1.44 s PCM crop ending with the five-class verifier. Merge at crop end "
             "with C161 and existing cooldown. Also union the short-window CTC result with the Q8 name-contrast "
             "and five-class joint confirmations before cooldown. Compare full address sentences in the three context "
             "streams for diagnosis only. GPU software reference; notification times exclude inference."))
        print(group,flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(dir=RESULTS) as tmp:
        directory = Path(tmp)
        main(directory)
        for path in directory.glob(f"{STEM}-*.json"):
            path.replace(RESULTS/path.name)
