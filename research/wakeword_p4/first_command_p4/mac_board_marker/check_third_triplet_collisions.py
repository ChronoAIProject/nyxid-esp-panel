"""Run the published identical-address collision comparison with the surviving triple cue."""
from pathlib import Path
from unittest.mock import patch

import bella_third_triplet as triple
import check_bella_headroom_collisions as checks

P, fit = triple.P, triple.fit
NAME = "multirate-before-nominal"
REPLAY = triple.OUT.with_name(f"4090-bella-nominal-first-{NAME}.json")
IDENTITY = triple.OUT.with_name("4090-bella-third-triplet-comparison.json")
OUT = triple.OUT.with_name("4090-bella-third-triplet-collisions.json")


def main():
    measured = P.read(REPLAY)
    cue = triple.wave(NAME)
    build = measured["build"]
    assert P.read(triple.OUT)["voices"][NAME] == P.sha(cue)
    assert measured["counts"]["rate"]["after"]==15 and measured["counts"]["synchronous"]["after"]==0
    P.save(IDENTITY, dict(fit=dict(cue_sha256=P.sha(cue), original_cue_sha256=P.sha(fit.early.WAV)), build=build,
                          sources={str(p):P.sha(p) for p in (triple.PROTO,triple.OUT,REPLAY,Path(__file__).resolve().relative_to(Path.cwd()))}))
    with patch.multiple(fit, WORK=triple.WORK/NAME, WAV=cue, OUT=IDENTITY), patch.multiple(checks, OUT=OUT, PROTO=triple.PROTO):
        checks.main()


if __name__ == "__main__":
    main()
