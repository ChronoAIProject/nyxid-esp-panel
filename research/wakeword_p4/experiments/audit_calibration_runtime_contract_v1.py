"""Check producer/packager metadata with an analytic fixture; emit a proposal only.

Run from the repo root with --training <original-v30-training.json> --output
<fresh-directory>. This writes a proposed patch and audit, never applies it.
No scores, models, microphone, speakers, board or training process are used.
"""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def result_call(tree):
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'calibrate')
    return next(x.value for x in function.body if isinstance(x, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'result' for t in x.targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    source_path = 'research/wakeword_p4/experiments/calibrate_recovered_new_development_v1.py'
    original = subprocess.check_output(['git', 'show', '21fa349:' + source_path]).decode()
    assert digest(original.encode()) == '92cf6f4517a5828741a26f027120502407ba557bff1262a108c5f26b0f0083da'
    needle = '                viable_operating_point=selected is not None,selected=selected,\n'
    assert original.count(needle) == 1
    patched = original.replace(needle, needle + '                validation_only=True,final_data_used=False,sliding_window=5,\n')
    old_tree, new_tree = ast.parse(original), ast.parse(patched)
    old_call, new_call = result_call(old_tree), result_call(new_tree)
    newkeys = {'validation_only', 'final_data_used', 'sliding_window'}
    assert not newkeys.intersection(k.arg for k in old_call.keywords)
    assert newkeys.issubset(k.arg for k in new_call.keywords)
    new_call.keywords = [k for k in new_call.keywords if k.arg not in newkeys]
    assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
    source = Path('research/panel_wake_runtime/prepare_runtime.py')
    spec = importlib.util.spec_from_file_location('runtime_contract', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training_bytes = args.training.read_bytes()
    assert digest(training_bytes) == '41d3a232920143d6bfa39b74fc1068f2675615f2df23ed599155b58a085312f9'
    training = json.loads(training_bytes)
    profile = json.loads(Path('research/panel_wake_runtime/plans/gpu-v30-runtime-v1.json').read_text())
    profile['calibration_schema'] = 'joint-development-v1'
    fixture = dict(role='Reserved Joe/new42-recording DEVELOPMENT calibration only',
        model_sha256=training['model_sha256'], execution_pins_sha256='fixture-only', source_sha256='fixture-only',
        candidate_report_sha256='fixture-only', grid=[], minimum_macro=.8, threshold=.91796875,
        viable_operating_point=True, selected=dict(eligible=True, threshold=.91796875), selected_synthetic_details={},
        acceptance_authorized=False, candidate_promoted=False,
        limits='Analytic packaging metadata fixture; not a real candidate.')
    assert set(fixture) == {k.arg for k in old_call.keywords}
    checks = []

    def run_case(name, value, expect):
        blob = json.dumps(value).encode()
        current = {**profile, 'training_report_sha256': digest(training_bytes), 'calibration_sha256': digest(blob)}
        try:
            module.validate_gpu_reports(current, training_bytes, blob)
            result = 'accepted'
        except (KeyError, ValueError) as exc:
            result = type(exc).__name__ + ': ' + str(exc)
        assert (result == 'accepted') == expect, (name, result)
        checks.append(dict(case=name, result=result))

    run_case('current producer output metadata rejected by existing packager', fixture, False)
    fixed = {**fixture, 'validation_only': True, 'final_data_used': False, 'sliding_window': 5}
    run_case('three explicit existing metadata fields restore compatibility', fixed, True)
    for key, value in [('validation_only', False), ('final_data_used', True), ('sliding_window', 3),
                       ('selected', None), ('viable_operating_point', False)]:
        run_case('reject ' + key, {**fixed, key: value}, False)
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile='a/' + source_path, tofile='b/' + source_path))
    report = dict(schema='mac-candidate-calibration-runtime-contract-audit-v1',
        source_sha256=digest(Path(__file__).read_bytes()), upstream_commit='21fa349',
        upstream_calibrator_sha256=digest(original.encode()), proposed_calibrator_sha256=digest(patched.encode()),
        patch_sha256=digest(patch.encode()), existing_packager_sha256=digest(source.read_bytes()),
        missing_metadata_fields=sorted(newkeys),
        entire_producer_ast_identical_after_removing_three_added_metadata_keywords=True,
        checks=checks, actual_candidate_scored=False, actual_candidate_packaged=False, training=False, firmware_modified=False,
        limits='Analytic metadata contract check using original authenticated v30 training metadata and synthetic '
            'selected-report metadata. No real recovered calibration exists yet. Does not prove candidate '
            'eligibility, report truth, inference or board acceptance. Proposed patch must be applied by GPU '
            'before future candidate freeze; current formal pins and live source untouched.')
    args.output.mkdir()
    (args.output / 'proposed.patch').write_text(patch)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
