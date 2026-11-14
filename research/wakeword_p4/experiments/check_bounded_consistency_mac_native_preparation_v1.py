"""Check prospective native identities/plan guards, without model execution."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace


BASE = Path(__file__).resolve().parents[1]
PREFIX = 'research/wakeword_p4/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    derivation_path = BASE / 'results/mac-bounded-consistency-native-replay-derivation-v1.json'
    derivation = json.loads(derivation_path.read_text())
    repo = BASE.parents[1]
    parent = repo / derivation['parent_path']
    derived = repo / derivation['derived_path']
    assert sha(parent) == derivation['parent_sha256']
    assert sha(derived) == derivation['derived_sha256']
    text = parent.read_text()
    for change in derivation['literal_identity_substitutions']:
        assert text.count(change['old']) == change['occurrences']
        text = text.replace(change['old'], change['new'])
    guard = """    expected = {(target, condition) for target in ['hey_willow', 'hey_nyx']
                for condition in ['raw_logit_control', 'bounded_logit_consistency']}
    assert len(arms) == 4 and {(c['target'], c['condition']) for c in arms} == expected
    assert formal['all_passed'], 'All four original formal arms must be complete'
"""
    text = text.replace('    assert len(arms) == 4\n', guard)
    text = text.replace('Exactly four existing fixed-final10000/fixed-grid models,',
        'Exactly four matched bounded-family fixed-final10000/fixed-grid models (raw_logit_control and bounded_logit_consistency),')
    assert text == derived.read_text()
    auditor = derivation['saved_array_auditor']
    text = (repo / auditor['parent_path']).read_text()
    assert sha(repo / auditor['parent_path']) == auditor['parent_sha256']
    for change in auditor['literal_identity_substitutions']:
        assert text.count(change['old']) == change['occurrences']
        text = text.replace(change['old'], change['new'])
    assert text == (repo / auditor['derived_path']).read_text()
    assert sha(repo / auditor['derived_path']) == auditor['derived_sha256']
    spec = importlib.util.spec_from_file_location('bounded_native_preparation', derived)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_source = module.source
    compile_name = 'results/4090-hey-broad-bounded-consistency-fixed-grid-compile-result-v1.json'
    formal_name = 'results/4090-hey-broad-bounded-consistency-formal-result-v1.json'
    arms, formal = [], []
    for target in ['hey_willow', 'hey_nyx']:
        for condition in ['raw_logit_control', 'bounded_logit_consistency']:
            label = target + '-' + condition
            arms.append(dict(target=target, condition=condition,
                path='.local/wakeword/constructed-only/' + label + '.tflite',
                sha256='0' * 64, bytes=123, final_h5_sha256='1' * 64))
            formal.append(dict(target=target, condition=condition, steps=10000,
                final_repeat=dict(weights_path='.local/wakeword/constructed-only/' + label + '.h5',
                                  weights_sha256='1' * 64)))
    fixture = {compile_name: dict(arms=arms), formal_name: dict(cases=formal, all_passed=True)}
    checks = []
    mutations = [
        ('complete_unique_constructed_four', lambda d: None, True),
        ('duplicate_model_arm', lambda d: d[compile_name]['arms'].__setitem__(3, copy.deepcopy(d[compile_name]['arms'][0])), False),
        ('missing_model_arm', lambda d: d[compile_name]['arms'].pop(), False),
        ('unexpected_condition', lambda d: d[compile_name]['arms'][0].update(condition='unplanned'), False),
        ('incomplete_formal', lambda d: d[formal_name].update(all_passed=False), False),
        ('not_final10000', lambda d: d[formal_name]['cases'][0].update(steps=9999), False),
        ('wrong_final_h5', lambda d: d[compile_name]['arms'][0].update(final_h5_sha256='2' * 64), False),
        ('duplicate_formal_case', lambda d: d[formal_name]['cases'].append(copy.deepcopy(d[formal_name]['cases'][0])), False),
    ]
    with tempfile.TemporaryDirectory(prefix='bounded-native-plan-constructed-') as temporary:
        for label, mutate, expected_success in mutations:
            data = copy.deepcopy(fixture)
            mutate(data)
            def source(ref, name):
                if name in data:
                    return json.dumps(data[name]).encode()
                if name == 'results/4090-hey-broad-bounded-consistency-fixed-grid-compile-plan-v1.json':
                    return b'{"constructed_only": true}'
                return real_source(ref, name)
            module.source = source
            module.PLAN = Path(temporary) / (label + '.json')
            args = SimpleNamespace(gpu_ref=a.gpu_ref, old_feature_payload=a.old_feature_payload,
                                   registered_fc_root=a.registered_fc_root)
            failure = None
            try:
                module.make_plan(args)
            except (AssertionError, ValueError, StopIteration) as error:
                failure = type(error).__name__
            assert (failure is None) == expected_success, label
            assert module.PLAN.exists() == expected_success, label
            checks.append(dict(case=label, expected_success=expected_success,
                               actual_success=failure is None, rejection=failure,
                               no_failed_plan_written=expected_success or not module.PLAN.exists()))
    report = dict(schema='mac-bounded-consistency-native-preparation-checks-v1',
        gpu_ref=subprocess.check_output(['git', 'rev-parse', a.gpu_ref], text=True).strip(),
        script_sha256=sha(__file__), derivation_sha256=sha(derivation_path),
        exact_declared_native_derivative=True, exact_literal_auditor_derivative=True,
        constructed_metadata_checks=checks, all_passed=True,
        limits='Constructed model/formal metadata only; existing consumed old feature bytes read for plan checks. No actual new H5/TFLite, interpreter, TensorFlow, audio, training, export, recognition or candidate admission. Temporary constructed plans are not actual model locks.')
    with a.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(a.report), constructed_checks=len(checks), all_passed=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--old-feature-payload', type=Path, required=True)
    parser.add_argument('--registered-fc-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
