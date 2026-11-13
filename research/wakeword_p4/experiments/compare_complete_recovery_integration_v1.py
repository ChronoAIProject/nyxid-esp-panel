"""Compare original, default-off and enabled real-trainer two-update captures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_complete_recovery_integration_v1 import BASE_SHA, RECOVERY_SHA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def same(left, right, names):
    for name in names:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'disabled', 'enabled', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    roots = [args.original, args.disabled, args.enabled]
    reports = [json.loads((root / 'integration-probe.json').read_text()) for root in roots]
    assert [r['trainer_sha256'] for r in reports] == [BASE_SHA, RECOVERY_SHA, RECOVERY_SHA]
    for report in reports:
        assert report['schema'] == 'complete-recovery-real-trainer-two-update-probe-v1'
        assert report['optimizer_updates'] == 2 and report['frozen_batchnorm_unchanged']
        assert report['all_model_variables_finite']
        assert not report['validation_invoked'] and not report['export_invoked'] and not report['acceptance_claim']
        assert [item['step'] for item in report['steps']] == [1, 2]
        assert np.isfinite([item['values'] for item in report['steps']]).all()
    assert len({r['capture_script_sha256'] for r in reports}) == 1
    assert len({r['initialization_inventory_sha256'] for r in reports}) == 1
    assert reports[0]['config_sha256'] == reports[1]['config_sha256']
    assert reports[0]['feature_metadata_hashes'] == reports[1]['feature_metadata_hashes'] == reports[2]['feature_metadata_hashes']
    helper_hashes = [{k: v for k, v in r['source_hashes'].items() if k != 'train_stream.py'} for r in reports]
    assert helper_hashes[0] == helper_hashes[1] == helper_hashes[2]
    assert reports[0]['final_model_values_sha256'] == reports[1]['final_model_values_sha256']
    assert reports[2]['final_model_values_sha256'] != reports[0]['final_model_values_sha256']
    destination = np.r_[16:24, 56:68]
    untouched = np.setdiff1d(np.arange(128), destination)
    fields = ('pre_x', 'pre_y', 'pre_weight', 'order', 'main_x', 'main_y', 'main_weight')
    details = []
    for step in (1, 2):
        batches = [arrays(root / f'observed-network-input-step{step}.npz') for root in roots]
        actual = [arrays(root / f'actual-network-input-step{step}.npz') for root in roots]
        original, disabled, enabled = batches
        same(original, disabled, fields)
        same(disabled, enabled, ('pre_y', 'pre_weight', 'order', 'main_y', 'main_weight'))
        assert original['pre_x'].shape == enabled['pre_x'].shape == (128, 150, 40)
        np.testing.assert_array_equal(disabled['pre_x'][untouched], enabled['pre_x'][untouched])
        np.testing.assert_array_equal(enabled['pre_x'][destination], enabled['recovered_x'])
        np.testing.assert_array_equal(enabled['pre_y'][destination, 0], [1] * 8 + [0] * 12)
        changed = np.flatnonzero(np.any(disabled['pre_x'] != enabled['pre_x'], axis=(1, 2)))
        assert set(changed) <= set(destination) and np.any(changed < 32) and np.any(changed >= 32)
        for observed, network in zip(batches, actual, strict=True):
            same(observed, network, ('main_x', 'main_y', 'main_weight'))
            assert network['pair_x'].shape == (40, 150, 40)
        same(actual[0], actual[1], tuple(actual[0]))
        same(actual[1], actual[2], ('pair_x', 'pair_y', 'pair_weight'))
        state = [r['steps'][step - 1] for r in reports]
        assert state[0] == state[1]
        assert state[0]['rng_states'] == state[2]['rng_states']
        if step == 1:
            # Before any weights diverge, the unchanged pair input must retain
            # exactly the same BCE/ranking loss despite changed main inputs.
            assert state[0]['values'][2:4] == state[2]['values'][2:4]
        details.append(dict(step=step, changed_pre_permutation_slots=changed.tolist(),
                            other_108_main_inputs_exact=True, all_pair40_inputs_exact=True,
                            all_captured_rng_states_exact=True,
                            main_bce_changed=state[0]['values'][1] != state[2]['values'][1]))
    report = dict(schema='complete-recovery-real-trainer-integration-comparison-v1',
                  script_sha256=sha(Path(__file__)),
                  input_report_sha256={name: sha(root / 'integration-probe.json')
                                       for name, root in zip(('original', 'disabled', 'enabled'), roots, strict=True)},
                  default_off_matches_original_two_updates_exactly=True,
                  initial_model_inventory_exact=True, labels_weights_permutation_exact=True,
                  recovered_saved_draws_reach_actual_optimizer=True,
                  final_weights_differ_with_recovery=True,
                  optimizer_updates_per_arm=2, steps=details, acceptance_claim=False,
                  limits='Two-update integration on actual training inputs only. No complete training, quality comparison, new-source/final evaluation or firmware promotion.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
