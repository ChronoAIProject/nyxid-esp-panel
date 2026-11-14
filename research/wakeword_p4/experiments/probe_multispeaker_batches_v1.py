"""Actual paired CPU batches; no TensorFlow, wake inference or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from multispeaker_batch_v1 import Sampler, cell, digest, BASE

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-multispeaker-batch-probe-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-batch-probe-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_plan():
    paths = [Path(__file__), Path(__file__).with_name('multispeaker_batch_v1.py'),
             Path(__file__).resolve().parents[1] / 'training/ambient_sampling.py',
             R / '4090-multispeaker-features-independent-audit-v1.json',
             R / '4090-multispeaker-coverage-features-v1.json',
             R / '4090-multispeaker-context309-initialization-v1.json',
             Path(__file__).with_name('multispeaker_feature_training_contract_v1.md')]
    for arm in ['narrow', 'broad']:
        for name in ['features.json', 'features.bin', 'human.json', 'ambient.json', 'target-provenance.json']:
            paths.append(BASE / arm / name)
    plan = dict(schema='multispeaker-batch-probe-plan-v1', files={str(p): sha(p) for p in paths},
                seed=726, batches=50, save_actual_first_batches=2,
                context_frames=309, expected_new_gib=.1,
                recipe='Canonical(engine,label,text,speed,variant) ordered pools, source_id tie break; uniform two-model probability and equal-size pools.32positive/48synthetic-negative/48stratified-original-training-ambient, weight1/20, no prefix negative or mining/masking. Positive whole-container jitter3..27; negative complete timeline from max309/lead-25 through end. One common uniform variate per variable-length endpoint; same permutation/ambient windows across arms.',
                limits='Actual CPU materialization only; source durations and native stochastic outcomes differ, matched lexical/environment cells are not identical waveforms. Zero model/gradient/optimizer calls.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN), batches=50)))


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, h in plan['files'].items():
        assert sha(path) == h, path
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(plan['expected_new_gib'])], check=True)
    samplers = {arm: Sampler(arm, plan['seed']) for arm in ['narrow', 'broad']}
    narrow, broad = samplers.values()
    assert [cell(r) for r in narrow.rows] == [cell(r) for r in broad.rows]
    assert narrow.ambient == broad.ambient
    assert np.array_equal(narrow.ambient_probabilities, broad.ambient_probabilities)
    OUT.mkdir(exist_ok=False)
    receipts = []
    with (OUT / 'records.jsonl').open('x') as log:
        for step in range(1, plan['batches'] + 1):
            actual = {arm: sampler.batch() for arm, sampler in samplers.items()}
            nd, bd = [actual[arm][1] for arm in samplers]
            assert json.dumps(narrow.rng.bit_generator.state) == json.dumps(broad.rng.bit_generator.state)
            assert len(nd) == len(bd) == 128
            for left, right in zip(nd, bd, strict=True):
                assert left['kind'] == right['kind'] and left['label'] == right['label']
                if left['kind'] == 'ambient':
                    assert left == right
                else:
                    assert left['cell'] == right['cell']
                    if left['label']:
                        assert left['end'] - narrow.by_index[left['index']]['keyword_frame'] == right['end'] - broad.by_index[right['index']]['keyword_frame']
            for arm, ((x, y, w), descriptors) in actual.items():
                assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
                assert np.count_nonzero(y) == 32 and np.count_nonzero(w == 20) == 96
                row = dict(step=step, arm=arm, descriptors=descriptors,
                           x_sha256=digest(x), y_sha256=digest(y), weights_sha256=digest(w))
                if step <= plan['save_actual_first_batches']:
                    path = OUT / f'{arm}-step{step:02}.npz'
                    np.savez(path, x=x, y=y, weights=w)
                    row.update(actual_tensor_path=str(path), actual_tensor_sha256=sha(path))
                log.write(json.dumps(row, separators=(',', ':')) + '\n')
                log.flush()
                receipts.append(row)
    result = dict(schema='multispeaker-batch-probe-v1', plan_sha256=sha(PLAN),
                  batches_per_arm=50, actual_windows_per_arm=6400,
                  matched_cells_labels_ambient_and_rng=True,
                  records_path=str(OUT / 'records.jsonl'), records_sha256=sha(OUT / 'records.jsonl'),
                  actual_tensor_archives=4, model_calls=0, optimizer_updates=0,
                  limits=plan['limits'])
    with (R / '4090-multispeaker-batch-probe-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run'])
    args = parser.parse_args()
    write_plan() if args.action == 'plan' else run()
