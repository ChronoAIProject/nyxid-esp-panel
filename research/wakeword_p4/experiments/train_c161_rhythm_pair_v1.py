"""Run matched fits with identical source order, histories and budget for the two duration arms."""
import argparse
import hashlib
from unittest.mock import patch

import numpy as np

import check_c161_signed_legacy_v1 as isolated
from fit_c161_history_gain_v1 import RESULTS, read, sha
import replay_c161_midlevel_v1 as replay
import replay_c161_vowel_name_pair_v1 as cohorts
import train_c161_midlevel_v1 as mid
import train_c161_protected_confuser_v1 as forward

PAIR = "4090-c161-rhythm-pair"


def train(stem, route):
    previous = mid.sources
    def augmented():
        yield from previous()
        ledger = read(RESULTS/f"{PAIR}-sources-v1.json")
        archive = RESULTS/f"{PAIR}-targets-v1.npz"
        assert sha(archive)==ledger["archive_sha256"]
        with np.load(archive) as z:
            for row in ledger["records"]:
                view=row["variants"][route]
                pcm=z[view["key"]]
                assert hashlib.sha256(pcm.tobytes()).hexdigest()==view["pcm_sha256"]
                yield dict(key=PAIR+"_"+row["key"],voice=row["voice"],positive=row["positive"],
                           text=row["text"],end=view["end"]),pcm
    with patch.multiple(mid, STEM=stem, sources=augmented):
        mid.main()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route",choices=("native","rhythm"))
    parser.add_argument("actions",nargs="+")
    args=parser.parse_args()
    stem=PAIR+"-"+args.route
    for action in args.actions:
        if action=="train":
            train(stem,args.route)
        elif action=="forward":
            with patch.object(forward,"STEM",stem):
                forward.forward_replay()
        elif action=="isolated":
            with patch.object(isolated,"STEM",stem):
                isolated.main()
        elif action=="transfer":
            with patch.object(replay,"STEM",stem):
                replay.transfer()
        else:
            with patch.object(cohorts,"STEM",stem):
                cohorts.run(action)
