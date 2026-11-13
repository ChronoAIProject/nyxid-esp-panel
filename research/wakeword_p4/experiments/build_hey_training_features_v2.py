"""Separate v2 packs from unchanged saved v1 waveforms and full allocations."""
import argparse
import json
from pathlib import Path

import build_hey_training_features_v1 as base
from hey_prefix_crop_v2 import legal_max

ORIGINAL_GEOMETRY=base.geometry
PLAN=base.ROOT/'4090-hey-training-feature-plan-v2.json'
DEST=Path('.local/wakeword/hey-paired-training-features-v2')


def geometry(row,timing):
    result=ORIGINAL_GEOMETRY(row,timing)
    # Common negatives still keep their original full-timeline crop rule.
    if row.get('target_id') in base.TARGETS:
        maximum=legal_max(result)
        result.update(original_fixed_jitter_supported=result['all_positive_jitters_have_prefix_support'],
                      positive_jitter_max=maximum,all_positive_jitters_have_prefix_support=True)
    return result


def configure():
    base.PLAN=PLAN;base.DEST=DEST;base.geometry=geometry


def make_plan():
    configure();base.make_plan()
    result=json.loads(PLAN.read_text())
    result.update(schema='hey-training-feature-plan-v2',
                  prior_experiment='v1 failed prefix admission; no v1 training',
                  positive_boundary='Same complete generated prefix+RIR; original jitter draw3..27 capped at geometry legal maximum, require every target-positive legal maximum>=3. No waveform changes/filtering.',
                  positive_crop_policy='hey-prefix-preserving-v2',
                  protocol_sha256=base.sha(Path(__file__).with_name('hey_paired_prefix_preserving_v2.md')))
    for name in ('build_hey_training_features_v2.py','hey_prefix_crop_v2.py','hey_paired_prefix_preserving_v2.md'):
        path=Path(__file__).with_name(name);result['files'][str(path)]=base.sha(path)
    PLAN.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(final_v2_plan_sha256=base.sha(PLAN),training_launched=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('write-plan','build'));a=p.parse_args()
    if a.action=='write-plan':make_plan()
    else:configure();base.build()
