"""Verify actual published formal/compile chains before the native model lock."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = 'research/wakeword_p4/'
PREFIX = ROOT + 'results/4090-hey-broad-bounded-consistency-'


def review(a):
    ref = subprocess.check_output(['git', 'rev-parse', a.gpu_ref], text=True).strip()
    cache = {}
    def read(path):
        if path not in cache:
            cache[path] = subprocess.check_output(['git', 'show', ref + ':' + path])
        return cache[path]
    digest = lambda path: hashlib.sha256(read(path)).hexdigest()
    names = ['formal-execution', 'formal-result', 'formal-independent-audit',
             'formal-evidence-scope', 'fixed-grid-compile-plan',
             'fixed-grid-compile-result', 'fixed-grid-constant-audit']
    paths = {name: PREFIX + name + '-v1.json' for name in names}
    data = {name: json.loads(read(path)) for name, path in paths.items()}
    for child, field, parent in [
        ('formal-result', 'execution_sha256', 'formal-execution'),
        ('formal-independent-audit', 'result_sha256', 'formal-result'),
        ('fixed-grid-compile-result', 'plan_sha256', 'fixed-grid-compile-plan'),
        ('fixed-grid-constant-audit', 'plan_sha256', 'fixed-grid-compile-plan'),
        ('fixed-grid-constant-audit', 'recovered_sha256', 'fixed-grid-compile-result')]:
        assert data[child][field] == digest(paths[parent])
    scope = data['formal-evidence-scope']
    assert scope['formal']['execution_sha256'] == digest(paths['formal-execution'])
    assert scope['formal']['actual_fresh_runs_per_arm'] == 1
    assert scope['formal']['full10000_fresh_model_optimizer_repeat_verified'] is False
    verified = {}
    for name in ['formal-execution', 'fixed-grid-compile-plan']:
        for path, expected in data[name]['files'].items():
            if path.startswith(ROOT) and path.endswith(('.py', '.md', '.json')):
                assert digest(path) == expected, path
                verified[path] = expected
    expected = {(target, condition) for target in ['hey_willow', 'hey_nyx']
                for condition in ['raw_logit_control', 'bounded_logit_consistency']}
    def indexed(rows):
        result = {(row['target'], row['condition']): row for row in rows}
        assert len(rows) == len(result) == 4 and set(result) == expected
        return result
    formal = indexed(data['formal-result']['cases'])
    audited = indexed(data['formal-independent-audit']['cases'])
    compiled = indexed(data['fixed-grid-compile-result']['arms'])
    templates = indexed(data['fixed-grid-compile-plan']['arms'])
    constants = indexed(data['fixed-grid-constant-audit']['arms'])
    assert data['formal-result']['all_passed'] and data['fixed-grid-constant-audit']['all_passed']
    records = []
    for key in sorted(expected):
        case, audit, arm, template, constant = [table[key] for table in
            [formal, audited, compiled, templates, constants]]
        final = case['final_repeat']
        assert case['steps'] == final['steps'] == audit['steps'] == 10000
        assert final['repeat'] == 0 and final['finite'] and final['frozen_bn_exact']
        assert all(audit[name] is True for name in ['original_initial_values_exact',
            'frozen_bn_16arrays_exact', 'all_batch_schedule_hashes_exact',
            'adam_iteration10000', 'all_periodic_checkpoints_present'])
        assert final['weights_sha256'] == audit['weights_sha256'] == arm['final_h5_sha256']
        assert data['fixed-grid-compile-plan']['files'][final['weights_path']] == final['weights_sha256']
        assert template['weights'] == final['weights_path']
        assert arm['template_sha256'] == template['template']['sha256']
        assert arm['original_grids_exact'] and arm['only_allowed_buffers_changed']
        assert arm['compiler_invocations'] == 1 and arm['converter_invocations'] == arm['model_invocations'] == 0
        assert len(arm['initializers']) == 6 and all(c['all_real_zero'] for c in arm['initializers'])
        assert constant['model_sha256'] == arm['sha256']
        checks = {c['index']: c for c in constant['checks']}
        assert len(checks) == len(constant['checks']) == len(arm['constants']) == 20
        for c in arm['constants']:
            check = checks[c['index']]
            assert check['sha256'] == c['code_sha256']
            assert check['buffer_exact'] and check['runtime_constant_exact']
        records.append(dict(target=key[0], condition=key[1], steps=10000,
            final_h5_sha256=final['weights_sha256'], model_sha256=arm['sha256'],
            model_bytes=arm['bytes'], declared_constant_checks=20))
    report = dict(schema='mac-bounded-consistency-complete-formal-compile-metadata-review-v1',
        gpu_ref=ref, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        receipts_sha256={name: digest(path) for name, path in paths.items()},
        public_source_metadata_bindings=verified, arms=records, all_metadata_checks_passed=True,
        limits='Published metadata/source-chain review only; original private weights, Adam, batch traces, TFLite buffers and runtime outputs not read on Mac. GPU independent-audit fields verified as published evidence, not independently re-executed. Formal one run/arm. No numerical, recognition/FAR, fresh-development or P4 qualification.')
    with a.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report=str(a.report), arms=len(records),
        public_bindings=len(verified), all_metadata_checks_passed=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--report', type=Path, required=True)
    review(parser.parse_args())
