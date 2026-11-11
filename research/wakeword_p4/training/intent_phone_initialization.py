"""Paired training-only neutral versus random phone-head initialization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from intent_phone_fit import unique_bank
from intent_phone_taps import fit
from phrase_ctc_readout import load_bank, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'initial-weights', 'parent-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh study directory')
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan['source_sha256'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Source differs: '+name)
    for name, digest in plan['parent_study_sha256'].items():
        if sha(args.parent_study/name) != digest:
            raise ValueError('Parent bytes differ: '+name)
    if (sha(args.initial_weights) != plan['initializer_sha256']
            or sha(args.features/'features.bin') != plan['feature_data_sha256']):
        raise ValueError('Actual initializer or features differ')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    raw, labels, rows = load_bank(args.bank, args.features, plan['compact_bank'])
    raw, labels, rows = unique_bank(raw, labels, rows)
    if rows != json.loads((args.parent_study/'study/rows.json').read_bytes()) or len(rows) != plan['unique_crops']:
        raise ValueError('Unique source crops differ')
    x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    results = {kind: fit(kind, x, labels, plan, args.initial_weights, args.output,
                        tap=setting['tap'], head_initialization=setting['initialization'])
               for kind, setting in plan['arm_settings'].items()}
    control, treatment = results[plan['control_arm']], results[plan['treatment_arm']]
    parent = json.loads((args.parent_study/'study/full-fit-phones.json').read_bytes())
    if any(control[a] != parent[b] for a, b in (
        ('encoder_weights_after_sha256', 'encoder_weights_after_sha256'),
        ('head_weights_after_sha256', 'head_weights_sha256'),
        ('draw_schedule_sha256', 'draw_schedule_sha256'), ('summary', 'summary'))):
        raise ValueError('Prior full-fit phone control does not replay')
    if (any(control[k] != treatment[k] for k in ('binary_weights_before_sha256',
            'encoder_weights_before_sha256', 'draw_schedule_sha256', 'head_parameters', 'parameters'))
            or control['head_weights_before_sha256'] == treatment['head_weights_before_sha256']):
        raise ValueError('Paired initializer-only comparison differs')
    report = {'plan_sha256': sha(args.plan), 'rows_sha256': sha(args.output/'rows.json'),
        'parent_deep_control_replayed': True,
        'arms': {kind: {'result': f'{kind}.json', 'result_sha256': sha(args.output/f'{kind}.json'),
                        'summary': value['summary'], 'fit_gate_passed': value['fit_gate_passed'],
                        'parameters': value['parameters'], 'initial_diagnostic': value['initial_diagnostic']}
                 for kind, value in results.items()},
        'advance': treatment['fit_gate_passed'], 'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, **report}), flush=True)


if __name__ == '__main__':
    main()
