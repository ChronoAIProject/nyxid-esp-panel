"""Check familiar and out-of-grammar first verbs on new matched synthetic dialogues."""
from pathlib import Path
import sys
from unittest.mock import patch

import check_c161_melo_address_grammar as melo

STEM = "4090-c161-address-verb-challenge"
PLAN = melo.RESULTS / f"{STEM}-plan.json"
ARCHIVE = melo.RESULTS / f"{STEM}-wav.zip"
WORK = Path(".local/wakeword/c161-address-verb/challenge")
CARRIERS = ("open the kitchen blinds.", "unlock the garden gate.")
DAILY = (("Nick opened the blinds this morning.", "I think the kitchen looks brighter."),
         ("Did Mick unlock the gate?", "Can you see the garden from here?"))


def generate():
    script = Path(__file__).with_name("check_c161_address_verb.py")
    old = melo.RESULTS/"4090-c161-address-verb.json"
    assert melo.sha(script) == melo.read(old)["script_sha256"]
    with patch.multiple(melo, PLAN=PLAN, AUDIO=ARCHIVE, WORK=WORK, CARRIERS=CARRIERS, DAILY=DAILY):
        melo.generate()


def score():
    import check_c161_address_grammar as old
    import check_c161_address_verb as new
    baseline = melo.RESULTS/f"{STEM}-grammar.json"
    if not baseline.exists():
        old.evaluate(old.live_inputs(PLAN,"challenge"),baseline,[PLAN])
    def streams():
        from select_c161_synthesized_cue_v2 import load_wav
        for src,r in zip(melo.read(PLAN)["records"],melo.read(baseline)["records"],strict=True):
            yield load_wav(src),src,r
    new.run(streams(),melo.RESULTS/f"{STEM}.json",[PLAN,baseline,melo.RESULTS/f"{new.STEM}.json"])


def willow(action):
    import check_c161_onset_names_willow as runner
    with patch.multiple(runner, challenge=sys.modules[__name__], __file__=str(Path(__file__).resolve())):
        sys.argv.insert(1,action)
        runner.main()


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "generate":
        generate()
    elif action == "score":
        score()
    else:
        willow(action)
