"""Compare the three real-input, two-update inherited-positive captures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from capture_inherited_positive_integration_v1 import BASE_SHA, RECOVERY_SHA
from compare_complete_recovery_integration_v1 import arrays, same


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'disabled', 'enabled', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    roots = [args.original, args.disabled, args.enabled]
    reports = [json.loads((r / 'integration-probe.json').read_text()) for r in roots]
    assert [r['trainer_sha256'] for r in reports] == [BASE_SHA, RECOVERY_SHA, RECOVERY_SHA]
    assert [r['inherited_positive_recovery_enabled'] for r in reports] == [False, False, True]
    assert len({r['complete_recovery_enabled'] for r in reports}) == 1
    for r in reports:
        assert r['schema'] == 'inherited-positive-real-trainer-two-update-probe-v1'
        assert r['optimizer_updates'] == 2 and r['frozen_batchnorm_unchanged']
        assert r['all_model_variables_finite'] and not r['validation_invoked'] and not r['export_invoked']
        assert [x['step'] for x in r['steps']] == [1, 2]
        assert np.isfinite([x['values'] for x in r['steps']]).all()
        assert r['feature_metadata_hashes']['features.json'] == '9db652f0769e49be6e924aadbfc3cd3725870b2eb558ea5b25b25481970d1d4b'
    assert len({r['capture_script_sha256'] for r in reports}) == 1
    assert len({r['initialization_inventory_sha256'] for r in reports}) == 1
    assert reports[0]['config_sha256'] == reports[1]['config_sha256']
    assert reports[0]['final_model_values_sha256'] == reports[1]['final_model_values_sha256']
    assert reports[2]['final_model_values_sha256'] != reports[0]['final_model_values_sha256']
    assert reports[0]['original_positive_sources_by_step'] == reports[1]['original_positive_sources_by_step'] == reports[2]['original_positive_sources_by_step']
    helpers = [{k: v for k, v in r['source_hashes'].items() if k != 'train_stream.py'} for r in reports]
    assert helpers[0] == helpers[1] == helpers[2]
    destination = np.arange(24, 32)
    other = np.setdiff1d(np.arange(128), destination)
    details = []
    for step in (1, 2):
        observed = [arrays(r / f'observed-network-input-step{step}.npz') for r in roots]
        actual = [arrays(r / f'actual-network-input-step{step}.npz') for r in roots]
        same(observed[0], observed[1], tuple(observed[0]))
        same(actual[0], actual[1], tuple(actual[0]))
        fields = ('pre_y', 'pre_weight', 'order', 'main_y', 'main_weight', 'initial_positive_x')
        same(observed[1], observed[2], fields)
        np.testing.assert_array_equal(observed[1]['pre_x'][other], observed[2]['pre_x'][other])
        np.testing.assert_array_equal(observed[2]['pre_x'][destination], observed[2]['initial_positive_x'][destination])
        np.testing.assert_array_equal(observed[2]['pre_y'][destination], np.ones((8, 1)))
        np.testing.assert_array_equal(observed[2]['pre_weight'][destination], np.ones(8))
        for capture, network in zip(observed, actual, strict=True):
            assert capture['initial_positive_x'].shape == (32, 150, 40)
            same(capture, network, ('main_x', 'main_y', 'main_weight'))
            assert network['pair_x'].shape == (40, 150, 40)
        same(actual[1], actual[2], ('pair_x', 'pair_y', 'pair_weight'))
        changed = np.flatnonzero(np.any(observed[1]['pre_x'] != observed[2]['pre_x'], axis=(1, 2)))
        assert len(changed) and set(changed) <= set(destination)
        states = [r['steps'][step - 1] for r in reports]
        assert states[0] == states[1]
        assert states[0]['rng_states'] == states[2]['rng_states']
        if step == 1:
            assert states[0]['values'][2:4] == states[2]['values'][2:4]
        sources = reports[2]['original_positive_sources_by_step'][step - 1]
        assert len(sources) == 32
        for source in sources:
            assert source['split'] == 'training' and source['label'] == 1
            assert 0 <= source['actual_crop_start_frame'] < source['actual_crop_stop_frame'] <= source['frames']
            assert source['actual_crop_stop_frame'] - source['actual_crop_start_frame'] == 150
        restored = [sources[int(i)] for i in destination]
        details.append(dict(step=step, changed_pre_permutation_slots=changed.tolist(),
                            other120_main_and_all_pair40_exact=True,
                            original8_draws_reach_actual_optimizer=True,
                            restored_intended_text_counts=dict(Counter(r['text'] for r in restored)),
                            restored_source_engine_counts=dict(Counter(r['engine'] for r in restored)),
                            source_ids_texts_crops_all_three_arms_exact=True))
    result = dict(schema='inherited-positive-real-trainer-integration-comparison-v1',
        source_sha256=sha(Path(__file__)), input_report_sha256={str(i): sha(r / 'integration-probe.json') for i, r in enumerate(roots)},
        complete_recovery_enabled=reports[0]['complete_recovery_enabled'],
        optimizer_updates_per_arm=2, default_off_matches_original_exactly=True,
        common_initialization=True, restored_original_draws_reach_optimizer=True,
        other120_main_and_all_pair40_exact=True, labels_weights_permutation_rng_exact=True,
        enabled_final_weights_differ=True, steps=details, acceptance_claim=False,
        limits='Actual two-update input integration only. Original8 are a mixed inherited source/text distribution; '
               'no command-only causal or quality claim, new trained candidate, threshold selection or promotion.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
