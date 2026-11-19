"""Generate new continuous Piper dialogue and run the official Willow reference on each stream."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import check_c161_piper_spellings as piper

STEM = "4090-c161-spelling-gap-dialogue"
WORK = Path(".local/wakeword/c161-spelling-gap-dialogue")
PLAN = piper.RESULTS / f"{STEM}-plan.json"
PACK = piper.RESULTS / f"{STEM}-pcm.zip"
RUNS = WORK / "willow"
CARRIERS = ("dim the light above the reading chair.", "set a reminder to water the basil.",
            "turn down the music in the hallway.")
DAILY = (("Mix the flour into the warm milk.", "The next visitor is due before noon.",
          "Did you save the receipt from the bakery?", "I can hear the television upstairs."),
         ("Nick left a note beside the kettle.", "The cake mixture is ready for the oven.",
          "Could you tell me when the rain will stop?", "Thanks for keeping an eye on the door."),
         ("The next song is one of my favourites.", "We should move these boxes after lunch.",
          "Max is bringing the folding chairs.", "I think the bus leaves in six minutes."))
ADDRESSES = ("Hey Nyx", "Hye Nyx", "Hey Nicks", "Hey Willow", "Hey Nick", "Hey Max")


def generate():
    with patch.multiple(piper, CARRIERS=CARRIERS, DAILY=DAILY, ADDRESSES=ADDRESSES,
                        PLAN=PLAN, ARCHIVE=PACK, WORK=WORK):
        piper.generate()
    plan = piper.read(PLAN)
    plan["role"] = "Post-fit development challenge using two previously used OpenSLR83 voices and new utterances."
    plan["comparison_commit"] = "2516e3851300b8d0373a014b1893390fb0dd405c"
    for path in (Path(__file__).resolve(), piper.RESULTS / "4090-c161-spelling-gap-training.json",
                 piper.RESULTS / "4090-c161-proposal-spellings-training.json"):
        plan["files"][str(path)] = piper.sha(path)
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")


def main(action):
    if action == "generate":
        generate()
        return
    import run_c161_willow_matched_scenes_v1 as runner
    with patch.multiple(runner, PLAN=PLAN, OUT=RUNS, __file__=str(Path(__file__).resolve())):
        if action == "willow":
            runner.willow()
        elif action == "case":
            runner.case()
        else:
            raise ValueError(action)


if __name__ == "__main__":
    main(sys.argv.pop(1))
