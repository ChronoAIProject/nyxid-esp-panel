"""Review two completed original Willow logs without opening model/data blobs.

Logged loss values describe the batch before its update, not an evaluation of
the saved final model. This does not change coefficients or admit candidates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


REF = 'dccec959abfe4f78b171f8295d0f9f2689e632a0'
ROOT = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(name):
    return subprocess.check_output(['git', 'show', REF + ':' + ROOT + name])


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def run(a):
    execution_name = 'results/4090-hey-broad-consistency-formal-execution-v1.json'
    execution_bytes = source(execution_name)
    execution = json.loads(execution_bytes)
    assert execution['mode'] == 'formal' and execution['steps'] == 10000
    assert execution['conditions'] == ['dual_supervised_control', 'logit_consistency']
    assert execution['repeats'] == 1
    sources = {ROOT + execution_name: digest(execution_bytes)}
    for name in ['run_hey_broad_consistency_v1.py', 'hey_broad_consistency_loss_v1.py',
                 'hey_broad_consistency_gradient_execution_v2.md',
                 'hey_broad_consistency_gradient_merge_v1.py',
                 'hey_broad_consistency_protocol_v1.md']:
        relative = ROOT + 'experiments/' + name
        data = source('experiments/' + name)
        expected, = [h for path, h in execution['files'].items()
                     if path == relative or path.endswith('/' + relative)]
        assert digest(data) == expected
        sources[relative] = expected
    records = {}
    for condition, coefficient in [('dual_supervised_control', 0), ('logit_consistency', 1)]:
        prefix = 'results/4090-hey-broad-consistency-hey_willow-' + condition
        training_bytes = source(prefix + '-training-original-v1.json')
        receipt_bytes = source(prefix + '-completed-receipt-v1.json')
        training, receipt = json.loads(training_bytes), json.loads(receipt_bytes)
        sources[ROOT + prefix + '-training-original-v1.json'] = digest(training_bytes)
        sources[ROOT + prefix + '-completed-receipt-v1.json'] = digest(receipt_bytes)
        assert digest(training_bytes) == receipt['original_training_sha256'] == receipt['copied_training_sha256']
        assert receipt['execution_sha256'] == digest(execution_bytes)
        assert receipt['target'] == training['target'] == 'hey_willow'
        assert receipt['condition'] == training['condition'] == condition
        assert receipt['steps'] == training['steps'] == 10000
        assert receipt['weights_sha256'] == training['weights_sha256']
        assert receipt['optimizer_sha256'] == training['optimizer_sha256']
        for field in ['original_initial_values_exact', 'frozen_bn_16arrays_exact',
                      'batch_schedule_10000_exact', 'adam_32arrays_finite_iteration10000']:
            assert receipt[field] is True
        assert training['finite'] is True and training['frozen_bn_exact'] is True
        assert training['repeat'] == 0 and training['parameter_delta_l2'] > 0
        history = training['history']
        assert [row['step'] for row in history] == [1, 2] + list(range(1000, 10001, 1000))
        for row in history:
            assert all(math.isfinite(value) for key, value in row.items()
                       if key not in ['batch_sha256', 'step'])
            assert row['loss'] == f32(row['supervised_loss'] + coefficient * row['consistency_loss'])
            if coefficient == 0:
                assert row['gradient_l2'] == row['supervised_gradient_l2']
        records[condition] = training
    control = records['dual_supervised_control']['history']
    treatment = records['logit_consistency']['history']
    assert [(r['step'], r['batch_sha256']) for r in control] == [(r['step'], r['batch_sha256']) for r in treatment]
    initial_fields = ['supervised_loss', 'consistency_loss',
                      'supervised_gradient_l2', 'consistency_gradient_l2']
    assert all(control[0][key] == treatment[0][key] for key in initial_fields)
    pairs = [dict(step=c['step'], batch_sha256=c['batch_sha256'],
                  control_supervised_loss=c['supervised_loss'],
                  treatment_supervised_loss=t['supervised_loss'],
                  control_consistency_loss=c['consistency_loss'],
                  treatment_consistency_loss=t['consistency_loss'])
             for c, t in zip(control, treatment, strict=True)]
    final_c, final_t = control[-1], treatment[-1]
    report = dict(
        schema='mac-completed-willow-consistency-metadata-review-v1', gpu_ref=REF,
        script_sha256=digest(Path(__file__).read_bytes()), source_sha256=sources,
        published_receipts_match_original_training_bytes=True,
        locked_policy_and_24_logged_loss_compositions_match=True,
        all12_logged_batch_identities_match=True,
        initial_component_losses_and_norms_match=True, paired_logged_batches=pairs,
        logged_update10000=dict(
            consistency_treatment_over_control=final_t['consistency_loss'] / final_c['consistency_loss'],
            supervised_treatment_over_control=final_t['supervised_loss'] / final_c['supervised_loss']),
        all_metadata_checks_passed=True,
        limits='Metadata/source review only. Original GPU receipt flags for initialization, BN, complete10000 schedules and Adam are checked for consistency, not independently re-proved from private tensors. Twelve published training-batch observations per arm, not aggregate loss, saved-final-model evaluation or recognition. Logs describe pre-update loss for that step. No checkpoint/parameter selection, coefficient adjustment, optimizer, export, fresh development, acceptance or P4 operation. Complete four-run formal and quality results remain pending.')
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_sha256', 'paired_logged_batches']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
