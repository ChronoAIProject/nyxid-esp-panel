"""Authenticate original eight probes and independently reduce their losses.

Uses Python scalar math, hashes original x/y/w against the prior batch receipts,
and never imports TensorFlow or executes the packaged producer/reducer sources.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np


REF = '3f656a2'
PREFIX = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tracked(ref, name):
    return subprocess.check_output(['git', 'show', ref+':'+name])


def close(left, right):
    # Only published float64 report arithmetic; no model/admission tolerance.
    return math.isclose(left, right, rel_tol=2e-12, abs_tol=1e-15)


def bce(logit, label):
    signed = logit if label == 0 else -logit
    return max(signed, 0.) + math.log1p(math.exp(-abs(signed)))


def probability(logit):
    return 1/(1+math.exp(-max(-8., min(8., logit))))


def run(args):
    receipt = json.loads(tracked(REF, PREFIX+'results/4090-actual-saturation-mac-handoff-v1.json'))
    archive = args.archive.read_bytes()
    assert len(archive) == receipt['bytes'] and digest(archive) == receipt['sha256']
    with tarfile.open(args.archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == receipt['exact_entries'] == 14
        assert len({member.name for member in members}) == len(members)
        for member in members:
            path = PurePosixPath(member.name)
            assert str(path) == member.name and not path.is_absolute() and '..' not in path.parts
            assert member.isfile() and 0 <= member.size < 2000000
        content = {member.name: tar.extractfile(member).read() for member in members}
    manifest_bytes = content['manifest.json']
    assert manifest_bytes == tracked(REF, PREFIX+'results/4090-actual-saturation-mac-handoff-manifest-v1.json')
    manifest = json.loads(manifest_bytes)
    assert len(manifest['files']) == receipt['payloads'] == 13
    assert set(content) == {'manifest.json'} | {item['name'] for item in manifest['files']}
    for item in manifest['files']:
        data = content[item['name']]
        assert len(data) == item['bytes'] and digest(data) == item['sha256']
        if not item['name'].endswith('.npz'):
            assert data == tracked(item['source_commit'], item['source_path'])
    plan_name = '4090-actual-raw-consistency-saturation-plan-v1.json'
    result_name = '4090-actual-raw-consistency-saturation-result-v1.json'
    reduction_name = '4090-actual-raw-consistency-saturation-independent-reduction-v1.json'
    plan, original, reduction = [json.loads(content[name]) for name in
                                 [plan_name, result_name, reduction_name]]
    assert original['plan_sha256'] == digest(content[plan_name])
    assert len(original['cases']) == len(reduction['cases']) == 8
    for name in [plan_name, result_name]:
        assert reduction['files'][PREFIX+'results/'+name] == digest(content[name])
    producer_path = PREFIX+'experiments/probe_actual_raw_consistency_saturation_v1.py'
    reducer_path = PREFIX+'experiments/reduce_actual_raw_consistency_saturation_v1.py'
    assert plan['files'][producer_path] == digest(content[Path(producer_path).name])
    assert reduction['files'][reducer_path] == digest(content[Path(reducer_path).name])
    historical_bytes = tracked(manifest['source_commit'], plan['original_result'])
    assert digest(historical_bytes) == plan['files'][plan['original_result']]
    historical = json.loads(historical_bytes)['cases']
    assert len(historical) == 8
    args.output_dir.mkdir(exist_ok=False)
    for name, data in content.items():
        with (args.output_dir/name).open('xb') as stream:
            stream.write(data)
    by_source = {item['source_path']: item for item in manifest['files']}
    seen, cases, aggregate_terms = set(), [], {}
    for case in original['cases']:
        key = case['target'], case['batch']
        assert key not in seen
        seen.add(key)
        item = by_source[case['arrays_path']]
        assert item['sha256'] == case['arrays_sha256']
        with np.load(args.output_dir/item['name'], allow_pickle=False) as data:
            saved = {name: data[name].copy() for name in data.files}
        assert set(saved) == {'x', 'y', 'w', 'float_logits', 'quant_logits', 'losses',
                              'repeat_float_logits', 'repeat_quant_logits', 'repeat_losses',
                              'original_saved_losses'}
        assert all(np.isfinite(value).all() for value in saved.values())
        assert saved['x'].shape == (128, 225, 40)
        prior, = [row for row in historical if (row['target'], row['batch']) == key]
        batch_hash = digest(b''.join(saved[name].tobytes() for name in ['x', 'y', 'w']))
        assert batch_hash == prior['batch_sha256']
        for name in ['float_logits', 'quant_logits', 'losses']:
            assert np.array_equal(saved[name], saved['repeat_'+name])
        losses = saved['losses']
        assert losses.shape == (3,) and losses.dtype == np.float32
        assert np.array_equal(losses, saved['original_saved_losses'])
        assert np.array_equal(losses, np.asarray(prior['losses'], np.float32))
        assert losses[2] == np.float32(losses[0]+losses[1])
        floating, quantized, labels, weights = [saved[name].reshape(-1).tolist() for name in
                                                ['float_logits', 'quant_logits', 'y', 'w']]
        assert len(floating) == len(quantized) == len(labels) == len(weights) == 128
        assert set(labels) <= {0, 1} and set(weights) <= {1, 20}
        terms = {name: [] for name in ['same_positive', 'same_negative', 'opposed_saturation', 'other']}
        gaps = {name: [] for name in terms}
        supervised, wrong = [], 0
        for f, q, label, weight in zip(floating, quantized, labels, weights, strict=True):
            magnitude = abs((f-q)/8)
            term = .5*magnitude*magnitude if magnitude <= 1 else magnitude-.5
            group = ('same_positive' if f >= 8 and q >= 8 else
                     'same_negative' if f <= -8 and q <= -8 else
                     'opposed_saturation' if (f >= 8 and q <= -8) or (q >= 8 and f <= -8)
                     else 'other')
            gap = abs(probability(f)-probability(q))
            if group.startswith('same_'):
                assert gap == 0
                wrong += int((group == 'same_positive' and label == 0) or
                             (group == 'same_negative' and label == 1))
            terms[group].append(term)
            gaps[group].append(gap)
            supervised.append((bce(f, label)+bce(q, label))*.5*weight)
        total = math.fsum(value for values in terms.values() for value in values)
        groups = {name: dict(count=len(values), raw_c_sum=math.fsum(values),
                            fraction_of_c=math.fsum(values)/total if total else 0,
                            nominal_probability_gap_max=max(gaps[name]) if values else None)
                  for name, values in terms.items()}
        published, = [row for row in reduction['cases'] if (row['target'], row['batch']) == key]
        assert sum(row['count'] for row in groups.values()) == 128
        assert wrong == published['same_side_nominal_saturated_wrong_labels']
        assert case['historical_losses_exact'] and published['historical_losses_exact']
        for name, group in groups.items():
            reference = published['groups'][name]
            assert group['count'] == reference['count']
            for field in ['raw_c_sum', 'fraction_of_c', 'nominal_probability_gap_max']:
                if group[field] is None:
                    assert reference[field] is None
                else:
                    assert close(group[field], reference[field]), (key, name, field)
        assert close(total/128, published['recomputed_float64_c'])
        assert float(losses[1]) == published['original_tf32_c']
        for name, values in terms.items():
            aggregate_terms.setdefault(case['target'], {}).setdefault(name, []).extend(values)
        cases.append(dict(target=case['target'], batch=case['batch'],
            original_batch_sha256=batch_hash, original_input_labels_weights_bound=True,
            original_historical_float32_losses_byte_exact=True, saved_repeats_exact=True,
            groups=groups, same_side_saturated_wrong_labels=wrong,
            scalar_float64_supervised_loss=math.fsum(supervised)/128,
            saved_tf32_supervised_loss=float(losses[0]),
            scalar_float64_consistency_loss=total/128,
            saved_tf32_consistency_loss=float(losses[1]), arrays_sha256=item['sha256']))
    assert seen == {(target, batch) for target in ['hey_willow', 'hey_nyx'] for batch in range(1, 5)}
    aggregate = {}
    for target, groups in aggregate_terms.items():
        total = math.fsum(value for values in groups.values() for value in values)
        aggregate[target] = {name: dict(count=len(values), raw_c_sum=math.fsum(values),
                                fraction_of_c=math.fsum(values)/total if total else 0)
                             for name, values in groups.items()}
        for name, values in aggregate[target].items():
            reference = reduction['aggregate'][target][name]
            assert values['count'] == reference['count']
            assert close(values['raw_c_sum'], reference['raw_c_sum'])
            assert close(values['fraction_of_c'], reference['fraction_of_c'])
    result = dict(schema='mac-actual-raw-saturation-independent-scalar-reduction-v1',
        script_sha256=digest(Path(__file__).read_bytes()), gpu_ref=REF,
        archive_sha256=digest(archive), archive_bytes=len(archive),
        manifest_sha256=digest(manifest_bytes), source_payloads_verified=13,
        plan_sha256=digest(content[plan_name]), gpu_result_sha256=digest(content[result_name]),
        gpu_reduction_sha256=digest(content[reduction_name]),
        original_gradient_diagnostic_result_sha256=digest(historical_bytes),
        cases=cases, aggregate=aggregate, samples=1024, all_checks_passed=True,
        model_invocations=0, optimizer_updates=0,
        limits='Original initialization probes only. All13 payloads/published sources authenticated; actual x/y/w byte hashes match earlier original batch receipts, losses match earlier reported float32 values and saved originals/repeats. New scalar math.fsum/log1p-exp BCE/Huber/nominal sigmoid decomposition independently agrees with published vector reduction. Does not regenerate source features, model outputs/gradients or unsaved historical logits, independently prove unchanged model variables, measure final-model saturation/actual full quantized-head equality, establish sole cause of35 failures, or qualify recognition/FAR/fresh development/acceptance/P4.')
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(args.report), all_checks_passed=True,
                          samples=1024, aggregate=aggregate)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
