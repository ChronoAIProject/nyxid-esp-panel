"""Compare frozen ASR-feature and microfrontend classifiers on complete dialogue streams."""
from pathlib import Path
import sys
from unittest.mock import patch

import check_c161_melo_address_grammar as melo

STEM = "4090-c161-onset-letters"
PLAN = melo.RESULTS/f"{STEM}-challenge-plan.json"
ARCHIVE = melo.RESULTS/f"{STEM}-challenge-wav.zip"
WORK = Path(".local/wakeword/c161-onset-letters/challenge")
NAMES = ("Hey Nyx","Hye Nyx","Hey Nicks","Hi Nicks","Hey Willow","Hey Next","Hi Next","Hey Nick","Hi Nick",
         "Hey Mick","Hi Nikki")
CARRIERS = ("dim the corner light.","move the meeting to Friday.")
DAILY = (("The next light is on the corner.","Nick repaired the broken lamp."),
         ("The next meeting starts on Friday.","Please send Mick a reminder."))


def generate():
    fit = melo.read(melo.RESULTS/f"{STEM}-fit.json")
    assert all(melo.sha(melo.RESULTS/f"{STEM}-{n}.npz")==v["weights_sha256"] for n,v in fit["models"].items())
    with patch.multiple(melo, PLAN=PLAN, AUDIO=ARCHIVE, WORK=WORK, NAMES=NAMES, CARRIERS=CARRIERS, DAILY=DAILY):
        melo.generate()


def score():
    import numpy as np
    import torch
    import check_c161_address_fifo as fifo
    import check_c161_address_grammar as grammar
    import check_c161_address_verb_challenge as verbs
    import train_c161_onset_letters as trained
    from check_c161_pocketsphinx_first_command import load_reader
    from fit_c161_history_gain_v1 import RESULTS, read, save, sha
    from check_c161_v5_command_ns import digest
    from score_c161_piper_spellings import annotate
    from select_c161_synthesized_cue_v2 import load_wav

    baseline = RESULTS/f"{STEM}-challenge-grammar.json"
    if not baseline.exists():
        grammar.evaluate(grammar.live_inputs(PLAN,"next"),baseline,[PLAN])
    def data():
        yield from fifo.inputs()
        for plan,comparison in ((verbs.PLAN,RESULTS/f"{verbs.STEM}-grammar.json"),(PLAN,baseline)):
            for src,ref in zip(read(plan)["records"],read(comparison)["records"],strict=True):
                yield load_wav(src),src,ref

    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    frozen,proc,spec = trained.teacher()
    fit = read(trained.FIT)
    nets = {}
    for mode,r in fit["models"].items():
        path = trained.weight_file(mode)
        assert sha(path) == r["weights_sha256"]
        net = trained.readout() if mode=="pretrained" else trained.Small(frozen.config.vocab_size)
        with np.load(path) as f:
            net.load_state_dict({k:torch.from_numpy(f[k]) for k in f})
        nets[mode] = net.cuda().eval()
    reader, wh = load_reader()
    records = []
    for pcm,src,prior in data():
        assert digest(pcm)==prior["pcm_sha256"]
        windows = trained.onset.observations(pcm)
        hits = {n:[] for n in nets}
        checks = []
        if windows:
            first = windows[0]
            prefix = trained.onset.observations(pcm[:first[1]])[0]
            assert first[:2]==prefix[:2] and np.array_equal(first[2],prefix[2])
            for i in range(0,len(windows),32):
                batch = windows[i:i+32]
                captures = [trained.cropped(pcm,end) for _,end,_ in batch]
                latent, letters = trained.encode(frozen,proc,captures)
                x = torch.from_numpy(np.stack([f for _,_,f in batch])).cuda().float().unsqueeze(1)/255
                with torch.inference_mode():
                    # Round the latent features to the storage precision used in the training cache.
                    scores = {n:(net(torch.from_numpy(latent.astype(np.float16)).cuda().float()) if n=="pretrained"
                             else net(x)[0]).softmax(dim=1).cpu().numpy() for n,net in nets.items()}
                greedy = proc.batch_decode(letters.argmax(axis=2))
                for j,(onset,end,patch_) in enumerate(batch):
                    for n,v in scores.items():
                        assert np.isfinite(v[j]).all()
                        if v[j,0] >= .5:
                            hits[n].append(end)
                    checks.append(dict(onset=onset,available=end,pcm_sha256=digest(captures[j]),
                                input_sha256=digest(patch_),greedy=greedy[j],prob={n:v[j].tolist() for n,v in scores.items()}))
        old = prior["events"]
        routes = dict(c161=old["c161"], grammar=old["extra_grammar"])
        for mode,v in hits.items():
            routes[mode] = grammar.merged([],v)
            routes["c161_"+mode] = grammar.merged(old["c161"],v)
        if "carrier" in src:
            annotated = annotate(src,pcm,routes,[],[],reader)
            for row in annotated["rows"]:
                del row["native_proposals"], row["expanded_proposals"]
        else:
            annotated = grammar.score_simple(src,routes)
        records.append(dict(cohort=prior["cohort"],key=prior["key"],pcm_sha256=digest(pcm),
                            checks=checks,events=routes,**annotated))
        if prior["cohort"]!="daily":
            print(prior["cohort"],prior["key"],flush=True)
    refs=[trained.FIT,trained.SOURCES,PLAN,baseline,verbs.PLAN,RESULTS/f"{verbs.STEM}-grammar.json",
          RESULTS/"4090-c161-address-grammar.json",RESULTS/"4090-c161-melo-address-grammar.json",
          RESULTS/"4090-c161-onset-names-melo-grammar.json"]
    save(RESULTS/f"{STEM}-evaluation.json",dict(sources={p.name:sha(p) for p in refs},script_sha256=sha(Path(__file__)),
         classes=trained.CLASSES,gpu=torch.cuda.get_device_name(),asr=wh,pretrained=spec,records=records))


def willow(action):
    import check_c161_onset_names_willow as runner
    # The reusable summarizer names the report by the challenge prefix.
    with patch.multiple(sys.modules[__name__],STEM=STEM+"-challenge"), patch.multiple(
            runner,challenge=sys.modules[__name__],__file__=str(Path(__file__).resolve())):
        sys.argv.insert(1,action)
        runner.main()


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action in ("generate","score"):
        {"generate":generate,"score":score}[action]()
    else:
        willow(action)
