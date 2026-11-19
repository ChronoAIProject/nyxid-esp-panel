"""Compare native class-distilled readouts with the frozen teacher and wake baseline."""
from pathlib import Path
import sys
from unittest.mock import patch

import check_c161_melo_address_grammar as melo
import check_c161_onset_letters as latest

STEM = "4090-c161-onset-evidence"
PLAN = melo.RESULTS / f"{STEM}-plan.json"
WORK = Path(".local/wakeword/c161-onset-evidence")
AUDIO = melo.RESULTS / f"{STEM}-wav.zip"
CARRIERS = ("close the bedroom door.", "read my next appointment.")
DAILY = (("I closed the bedroom door.", "Nick lives next door."),
         ("My next appointment is tomorrow.", "Please read the next message."))


def generate():
    fit = melo.read(melo.RESULTS / f"{STEM}-fit.json")
    assert all(melo.sha(melo.RESULTS / f"{STEM}-{k}.npz") == v["weights_sha256"] for k,v in fit["models"].items())
    with patch.multiple(melo, PLAN=PLAN, AUDIO=AUDIO, WORK=WORK, NAMES=latest.NAMES, CARRIERS=CARRIERS, DAILY=DAILY):
        melo.generate()


def score():
    import numpy as np
    import torch
    import check_c161_address_grammar as grammar
    import check_c161_address_verb_challenge as verbs
    import check_c161_address_fifo as fifo
    import train_c161_onset_evidence as fit
    from check_c161_pocketsphinx_first_command import load_reader
    from check_c161_v5_command_ns import digest
    from fit_c161_history_gain_v1 import RESULTS, read, save, sha
    from score_c161_piper_spellings import annotate
    from select_c161_synthesized_cue_v2 import load_wav

    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    base = fit.base
    reader, identity = load_reader()
    frozen, proc, teacher_identity = base.teacher()
    previous = read(base.FIT)["models"]
    info = read(fit.FIT)
    teacher = fit.load(base.readout(), base.weight_file("pretrained"), previous["pretrained"]["weights_sha256"])
    nets = {k:fit.load(base.Small(info["alphabet"]), fit.weight_file(k), v["weights_sha256"]) for k,v in info["models"].items()}
    nets["small"] = fit.load(base.Small(info["alphabet"]), base.weight_file("small"), previous["small"]["weights_sha256"])
    path = RESULTS / f"{latest.STEM}-evaluation.json"
    old = {r["pcm_sha256"]:r for r in read(path)["records"]}
    baseline = RESULTS / f"{STEM}-grammar.json"
    if not baseline.exists():
        grammar.evaluate(grammar.live_inputs(PLAN,"appointment"), baseline, [PLAN])

    def inputs():
        for pcm, src, prior in fifo.inputs():
            yield pcm, src, old.pop(prior["pcm_sha256"])
        for p in (verbs.PLAN, latest.PLAN):
            for src in read(p)["records"]:
                yield load_wav(src), src, old.pop(src["pcm_sha256"])
        assert not old
        for src, prior in zip(read(PLAN)["records"],read(baseline)["records"],strict=True):
            yield load_wav(src), src, prior

    results = []
    for pcm, src, ref in inputs():
        assert digest(pcm) == ref["pcm_sha256"]
        fresh = ref["cohort"] == "appointment"
        modes = list(nets) + ["pretrained"] if fresh else list(info["models"])
        hits = {k:[] for k in modes}
        windows = base.onset.observations(pcm)
        checks = []
        if not fresh:
            assert [(o,e,digest(a)) for o,e,a in windows] == [(v["onset"],v["available"],v["input_sha256"]) for v in ref["checks"]]
        for i in range(0,len(windows),32):
            batch = windows[i:i+32]
            x = torch.from_numpy(np.stack([a for _,_,a in batch])).cuda().float().unsqueeze(1)/255
            with torch.inference_mode():
                probs = {k:nets[k](x)[0].softmax(dim=1).cpu().numpy() for k in modes if k in nets}
                if fresh:
                    latent, _ = base.encode(frozen,proc,[base.cropped(pcm,end) for _,end,_ in batch])
                    pred = teacher(torch.from_numpy(latent.astype(np.float16)).cuda().float())
                    probs["pretrained"] = pred.softmax(dim=1).cpu().numpy()
            for j,(onset,end,a) in enumerate(batch):
                for k,v in probs.items():
                    assert np.isfinite(v[j]).all()
                    if v[j,0] >= .5:
                        hits[k].append(end)
                checks.append(dict(onset=onset,available=end,input_sha256=digest(a),prob={k:v[j].tolist() for k,v in probs.items()}))
        events = ref["events"]
        routes = dict(c161=events["c161"],grammar=events["extra_grammar" if fresh else "grammar"])
        if not fresh:
            routes.update({k:events["c161_"+k] for k in ("small","pretrained")})
        routes.update({k:grammar.merged(events["c161"], times) for k,times in hits.items()})
        if "carrier" in src:
            ann = annotate(src,pcm,routes,[],[],reader)
            for row in ann["rows"]:
                del row["native_proposals"], row["expanded_proposals"]
        else:
            ann = grammar.score_simple(src,routes)
        results.append(dict(cohort=ref["cohort"],key=ref["key"],pcm_sha256=ref["pcm_sha256"],events=routes,checks=checks,**ann))
        if ref["cohort"] != "daily":
            print(ref["cohort"],ref["key"],flush=True)
    save(RESULTS/f"{STEM}-evaluation.json",dict(records=results, asr=identity, pretrained=teacher_identity,
         sources={p.name:sha(p) for p in [fit.FIT,base.FIT,PLAN,baseline,path,Path(__file__)]},
         policy="Each named auxiliary path merged with original C161 at frozen threshold and cooldown. Standard 2 s recording and unchanged suffix scoring."))


if __name__ == "__main__":
    action = sys.argv[1]
    if action in ("case","willow","summarize"):
        import check_c161_onset_names_willow as runner
        with patch.multiple(runner,challenge=sys.modules[__name__],__file__=str(Path(__file__).resolve())):
            runner.main()
    else:
        {"generate":generate,"score":score}[action]()
