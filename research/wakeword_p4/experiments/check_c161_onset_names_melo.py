"""Render new paired development carriers after freezing the onset classifier."""
from pathlib import Path
from unittest.mock import patch

import check_c161_melo_address_grammar as melo

STEM = "4090-c161-onset-names-melo"
PLAN = melo.RESULTS / f"{STEM}-plan.json"
ARCHIVE = melo.RESULTS / f"{STEM}-wav.zip"
WORK = Path(".local/wakeword/c161-onset-names/melo")
CARRIERS = ("save the shopping list.", "turn the hallway light off.")
DAILY = (("Nick saved the last photo.", "Can you see my shopping list?"),
         ("Mick took the later train.", "The hallway looks bright today."))


def generate():
    fit = melo.RESULTS / "4090-c161-onset-names-fit.json"
    weights = melo.RESULTS / "4090-c161-onset-names-weights.npz"
    assert melo.sha(weights) == melo.read(fit)["weights_sha256"]
    with patch.multiple(melo, PLAN=PLAN, AUDIO=ARCHIVE, WORK=WORK, CARRIERS=CARRIERS, DAILY=DAILY):
        melo.generate()


if __name__ == "__main__":
    generate()
