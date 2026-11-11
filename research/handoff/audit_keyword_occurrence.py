"""Independently reconstruct the training-only keyword occurrence diagnostic."""
import argparse
from collections import defaultdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_compact_study import require, sha


def probability_domain(logits):
    x = np.asarray(logits, np.float64)
    p = np.exp(x-x.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    state = np.zeros((len(p), 3)); state[:, 0] = 1
    for frame in p.transpose(1, 0, 2):
        transitions = np.zeros((len(p), 3, 3))
        transitions[:, 0, 0] = frame[:, [0, 2, 3, 4, 5]].sum(-1)
        transitions[:, 0, 1] = frame[:, 1]
        transitions[:, 1, 0] = frame[:, 3:].sum(-1)
        transitions[:, 1, 1] = frame[:, [0, 1]].sum(-1)
        transitions[:, 1, 2] = frame[:, 2]
        transitions[:, 2, 2] = 1
        state = np.einsum('bi,bij->bj', state, transitions)
    return state[:, 2]


def audit(args):
    root = args.artifacts.resolve(); output = root/'diagnostic'
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'launch.json').read_bytes())
    committed = subprocess.check_output(['git', 'show', launch['checkout_revision']+
        ':research/wakeword_p4/training/plans/keyword-occurrence-training-diagnostic-v1.json'], cwd=args.repository)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256'],
            'Plan differs from committed launch')
    require(sha(output/'plan.json') == sha(root/'plan.json'), 'Executed plan differs')
    for name, digest in plan['source_sha256'].items():
        require(sha(root/'source'/name) == launch['source_sha256'][name] == digest,
                'Executed source differs: '+name)
    command = launch['command']
    require(Path(command[1]).resolve() == root/'source/diagnose_keyword_occurrence.py'
            and command[2::2] == ['--plan', '--model', '--features', '--output'], 'Executed command differs')
    for name, expected in [('--plan', root/'plan.json'), ('--model', args.model),
                           ('--features', args.features), ('--output', output)]:
        require(Path(command[command.index(name)+1]).resolve() == expected.resolve(), 'Input path differs: '+name)
    require(sha(args.model) == plan['model_sha256']
            and sha(args.features/'features.json') == plan['feature_index_sha256']
            and sha(args.features/'features.bin') == plan['feature_data_sha256'], 'Actual model/features differ')
    result = json.loads((output/'result.json').read_bytes())
    require(result['plan_sha256'] == sha(root/'plan.json') and result['model_sha256'] == sha(args.model)
            and result['accepted'] is False, 'Result binding differs')
    rows = json.loads((args.features/'features.json').read_bytes())
    by_key = {(r['path'], r['variant']): r for r in rows}
    reserved = {r['sha256'] for r in rows if r['split'] != 'training'}
    # Independent CTC implementation: TensorFlow's dense-label loss for the
    # exact event; probability-domain transition matrices for occurrence.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    maximum_exact_delta = maximum_occurrence_delta = 0.
    grouped, steps, arrays = defaultdict(list), 0, {}
    for item, report in zip(plan['sources'], result['sources'], strict=True):
        row = by_key[item['path'], item['variant']]
        require(row['split'] == 'training' and row['sha256'] not in reserved
                and all(row[k] == item[k] for k in ('sha256', 'engine', 'text', 'label'))
                and report['source'] == item, 'Training source identity differs')
        path = output/report['array']; require(sha(path) == report['array_sha256'], 'Trace differs')
        with np.load(path, allow_pickle=False) as trace:
            p, logits = trace['scalar'], trace['logits']
            count = row['frames']//3
            require(p.shape == (count,) and logits.shape == (count, 67, 6)
                    and np.isfinite(logits).all() and np.isfinite(p).all()
                    and np.all((p >= 0) & (p <= 1)), 'Trace geometry/value differs')
            lengths = np.full(count, 2, np.int32)
            losses = tf.nn.ctc_loss(np.tile([1, 2], (count, 1)), logits, lengths,
                np.full(count, 67, np.int32), logits_time_major=False, blank_index=0)
            expected_exact = np.exp(-np.asarray(losses, np.float64))
            expected_occurrence = probability_domain(logits)
            exact_delta = float(np.abs(expected_exact-trace['exact']).max())
            occurrence_delta = float(np.abs(expected_occurrence-trace['occurrence']).max())
            require(exact_delta <= 1e-5 and occurrence_delta <= 1e-12, 'Independent CTC probability differs')
            maximum_exact_delta = max(maximum_exact_delta, exact_delta)
            maximum_occurrence_delta = max(maximum_occurrence_delta, occurrence_delta)
            values = {'scalar': p, 'exact_product': p*trace['exact'], 'occurrence_product': p*trace['occurrence']}
            scores = {name: np.convolve(v, np.ones(5)/5, mode='valid') for name, v in values.items()}
            times = (3*(np.arange(count-4)+4)+5)*.01
            end = (row['keyword_frame']+2)*.01 if row['label'] else None
            valid = (times >= end-.1) & (times <= end+.8) if end is not None else np.ones(len(times), bool)
            require(np.array_equal(times, trace['times']) and end == report['keyword_end'], 'Timebase differs')
            for name, value in scores.items():
                require(np.array_equal(value, trace[name+'_smoothed'])
                        and float(value[valid].max()) == report['maxima'][name], 'Score accounting differs')
            peak = np.flatnonzero(valid)[int(np.argmax(scores['scalar'][valid]))]
            tokens = [int(k) for k, _ in groupby(logits[peak+4].argmax(-1)) if k != 0]
            at_peak = {'time': float(times[peak]), 'greedy_tokens': tokens, 'greedy_exact': tokens == [1, 2],
                'greedy_contains': any(tokens[i:i+2] == [1, 2] for i in range(len(tokens)-1)),
                'exact': float(trace['exact'][peak+4]), 'occurrence': float(trace['occurrence'][peak+4])}
            require(at_peak == report['at_scalar_peak'] and report['steps'] == count, 'Peak/greedy accounting differs')
        grouped[item['cohort']].append(report); steps += count; arrays[path.name] = sha(path)
    summary = {name: {'sources': len(values),
        'median_maxima': {score: float(np.median([r['maxima'][score] for r in values])) for score in scores},
        'at_scalar_peak_greedy_exact': sum(r['at_scalar_peak']['greedy_exact'] for r in values),
        'at_scalar_peak_greedy_contains': sum(r['at_scalar_peak']['greedy_contains'] for r in values),
        'reference_0_97_views': {score: sum(r['maxima'][score] >= .97 for r in values) for score in scores}}
        for name, values in grouped.items()}
    require(summary == result['cohorts'], 'Cohort accounting differs')
    neutral = float(probability_domain(np.zeros((1, 67, 6)))[0])
    require(abs(neutral-result['neutral_67_step_logits']['occurrence']) <= 1e-12, 'Neutral prior differs')
    return {'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root/'plan.json'),
        'result_sha256': sha(output/'result.json'), 'frozen_source_plan_model_and_feature_bytes_match': True,
        'source_count': len(result['sources']), 'raw_streaming_steps': steps,
        'maximum_tensorflow_exact_probability_delta': maximum_exact_delta,
        'maximum_matrix_occurrence_probability_delta': maximum_occurrence_delta,
        'all_cohorts_traces_maxima_timebases_and_greedy_sequences_reconstructed': True,
        'trace_sha256': arrays,
        'limits': 'Supplementary read-only reconstruction of optimization-seen training diagnostics. Exact probability uses an independent float32 CTC loss (1e-5 absolute tolerance); occurrence uses float64 transition matrices (1e-12). No quality acceptance, fitting, new model inference, held-out/audio/board access.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'model', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); result = audit(args)
    with args.output.open('x') as out:out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'trace_sha256'}))


if __name__ == '__main__':
    main()
