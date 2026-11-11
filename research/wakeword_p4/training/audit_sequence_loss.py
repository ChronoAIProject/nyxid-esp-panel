"""Audit saved stable-loss values/gradients with independent NumPy float64 math."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def oracle(labels, logits):
    logits = np.asarray(logits, dtype=np.float64)
    log_positive = -np.logaddexp(0, -logits)
    log_negative = -np.logaddexp(0, logits)
    lp = np.logaddexp.reduce(log_positive, axis=1)
    ln = np.logaddexp.reduce(log_negative, axis=1)
    loss = labels*(np.log(logits.shape[1])-lp)+(1-labels)*(np.log(logits.shape[1])-ln)
    positive_gradient = -np.exp(log_positive-lp[:, None]+log_negative)
    negative_gradient = np.exp(log_negative-ln[:, None]+log_positive)
    return loss, labels[:, None]*positive_gradient+(1-labels[:, None])*negative_gradient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.study/'result.json').read_bytes())
    assert sha((args.study/'plan.json').read_bytes()) == report['plan_sha256']
    assert sha((args.study/'arrays.npz').read_bytes()) == report['arrays_sha256']
    for name, expected in report['source_sha256'].items():
        assert sha((Path(__file__).parent/name).read_bytes()) == expected
    actual, scalar_error, probability_error = {}, {'loss': 0., 'gradient': 0.}, {}
    for case in report['scalar_cases']:
        for mode in ('terminal', 'five_mean'):
            logits = np.asarray([case['logits']], dtype=np.float64)
            if mode == 'terminal':
                logits = logits[:, -1:]
            expected_loss, expected_gradient = oracle(np.asarray([case['label']]), logits)
            saved = case[mode]
            gradient = np.asarray(saved['stable_gradient'])
            if mode == 'terminal':
                assert np.all(gradient[:, :-1] == 0)
                gradient = gradient[:, -1:]
            loss_error = float(np.max(np.abs(expected_loss-saved['stable_loss'])))
            gradient_error = float(np.max(np.abs(expected_gradient-gradient)))
            scalar_error['loss'] = max(scalar_error['loss'], loss_error)
            scalar_error['gradient'] = max(scalar_error['gradient'], gradient_error)
            assert loss_error < 1e-4 and gradient_error < 1e-5
    with np.load(args.study/'arrays.npz', allow_pickle=False) as arrays:
        labels = arrays['labels'].astype(np.float64)
        assert labels.shape == (800,) and set(labels) == {0., 1.}
        for model in report['summary']:
            actual[model] = {}
            logits = arrays[model+'_logits']
            probabilities = arrays[model+'_probabilities']
            oracle_probability = np.exp(-np.logaddexp(0, -logits.astype(np.float64)))
            probability_error[model] = float(np.max(np.abs(probabilities-oracle_probability)))
            np.testing.assert_allclose(probabilities, oracle_probability, atol=2e-7, rtol=0)
            for mode in ('terminal', 'five_mean'):
                prefix = model+'_'+mode+'_'
                z = logits[:, -1:] if mode == 'terminal' else logits
                expected_loss, expected_gradient = oracle(labels, z)
                stable_gradient = arrays[prefix+'stable_gradient']
                if mode == 'terminal':
                    assert np.all(stable_gradient[:, :-1] == 0)
                    stable_gradient = stable_gradient[:, -1:]
                loss_error = float(np.max(np.abs(expected_loss-arrays[prefix+'stable_loss'])))
                gradient_error = float(np.max(np.abs(expected_gradient-stable_gradient)))
                assert loss_error < 1e-5 and gradient_error < 1e-6
                old_gradient = arrays[prefix+'old_gradient']
                all_stable_gradient = arrays[prefix+'stable_gradient']
                old_zero = np.max(np.abs(old_gradient), axis=1) == 0
                signal = np.max(np.abs(all_stable_gradient), axis=1)
                affected = old_zero & (signal > 1e-8)
                assert int(affected.sum()) == report['summary'][model][mode]['total']['old_zero_gradient_stable_signal']
                actual[model][mode] = {'rows': len(labels),
                    'loss_oracle_max_error': loss_error, 'gradient_oracle_max_error': gradient_error,
                    'old_zero_gradient_stable_gt_1e_minus_8': int(affected.sum()),
                    'old_zero_gradient_stable_gt_1e_minus_6': int(np.count_nonzero(old_zero & (signal > 1e-6))),
                    'affected_wrong_decision_rows': int(np.count_nonzero(affected & (expected_loss >= np.log(2)))),
                    'affected_max_stable_loss': float(np.max(arrays[prefix+'stable_loss'][affected])) if affected.any() else None,
                    'affected_max_abs_stable_gradient': float(np.max(signal[affected])) if affected.any() else None}
    result = {'scope': __doc__, 'source_result_sha256': sha((args.study/'result.json').read_bytes()),
        'audit_source_sha256': sha(Path(__file__).read_bytes()), 'scalar_oracle_max_error': scalar_error,
        'probability_float64_oracle_max_error': probability_error,
        'actual_training_diagnostic': actual,
        'decision': 'Artificial high-confidence wrong logits expose a real numerical hazard, but these800 old training draws show only tiny lost gradients on confidently correct examples. Do not attribute recognition failure to clipping or retrain on this evidence.',
        'limits': ['Not independent model inference, original probability-BCE gradient replay, or verification of reported per-engine groups.',
                   'No optimizer, validation/new/final scoring, hardware, acoustic or recognition acceptance.']}
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
