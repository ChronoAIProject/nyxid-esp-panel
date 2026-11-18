"""Replay v5 with fixed controls, an independent room seed series, and the C161 baseline."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import check_c161_coded_cue_aligned_v4 as aligned
import check_c161_coded_cue_band_v4 as band
import check_c161_coded_cue_v4 as replay
from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import RESULTS, read, sha
from generate_c161_coded_cue_v5 import COEFF, STEM, WAV
from select_c161_synthesized_cue_v2 import load_wav

HEADER=RESULTS.parent/"portable/c161_coded_cue_v5.h"


if __name__=="__main__":
    spec=read(RESULTS/f"{STEM}-source.json")
    assert sha(COEFF)==spec["coefficients_sha256"]
    cue=load_wav(dict(path=WAV,sha256=spec["wav_sha256"]))
    with tempfile.TemporaryDirectory() as temp:
        lib=binding(Path(temp)/"v5.so",coded="v5")
        with patch.multiple(replay,STEM=STEM,WAV=WAV,HEADER=HEADER,COEFF=COEFF,TARGET="v5"), \
             patch.multiple(band,STEM=STEM,WAV=WAV,HEADER=HEADER,COEFF=COEFF),patch.multiple(aligned,STEM=STEM,HEADER=HEADER):
            for group in sys.argv[1:]:
                if group in ("primed","jenny","melo"):
                    aligned.background(lib,group)
                elif group=="fresh":
                    band.fresh(lib,cue,target="v5",seed_base=316523)
                elif group=="merged":
                    aligned.combine(cue)
                else:
                    replay.run(lib,group,cue)
