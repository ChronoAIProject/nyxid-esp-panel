"""Replay the one-output mid-level model on unchanged speech and broadcast inputs."""
import hashlib
import sys
from unittest.mock import patch

import numpy as np

from audit_c161_history_paths_v1 import arbitrate, decode, times
from check_c161_bounded_transfer_v1 import sources
import check_c161_signed_legacy_v1 as isolated
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
import replay_c161_vowel_name_pair_v1 as cohorts
import train_c161_protected_confuser_v1 as forward
from train_c161_midlevel_v1 import STEM
import verify_c161_ambient_stream_v1 as live


def transfer():
    model = read(RESULTS/f"{STEM}-training-v1.json")["reports"][0]
    path = RESULTS/model["path"]
    assert sha(path)==model["sha256"]
    native = binding()
    baseline = RESULTS/"4090-c161-bounded-readout-transfer-scores-v1.npz"
    assert sha(baseline)==read(RESULTS/"4090-c161-bounded-readout-transfer-v1.json")["scores_sha256"]
    arrays,rows={},[]
    with np.load(baseline) as old, patch.object(live,"MODEL",path):
        for item,pcm in sources():
            key=item["group"]+"_"+item["key"]
            assert hashlib.sha256(pcm.tobytes()).hexdigest()==item["pcm_sha256"]
            values,clock,emitted,suppressed=live.run_stream(native,pcm)
            clock=np.asarray(clock)
            np.testing.assert_array_equal(clock,old[key+"_clock"])
            arrays[key]=np.asarray(values,np.uint8)
            primary,auxiliary=[decode(v,clock) for v in arrays[key]]
            assert arbitrate(primary,auxiliary)==(emitted,suppressed)
            rows.append(dict(**item,primary=times(primary),auxiliary=times(auxiliary),pair=times(emitted)))
    out=RESULTS/f"{STEM}-transfer-scores-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-transfer-v1.json",dict(model_sha256=sha(path),clock_source_sha256=sha(baseline),
         scores_sha256=sha(out),records=rows))


if __name__=="__main__":
    for action in sys.argv[1:]:
        if action=="isolated":
            with patch.object(isolated,"STEM",STEM):
                isolated.main()
        elif action=="forward":
            with patch.object(forward,"STEM",STEM):
                forward.forward_replay()
        elif action=="transfer":
            transfer()
        else:
            with patch.object(cohorts,"STEM",STEM):
                cohorts.run(action)
