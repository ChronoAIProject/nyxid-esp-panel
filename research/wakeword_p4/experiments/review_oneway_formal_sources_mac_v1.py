"""Read-only review of published one-way training and validation source bindings.

This inspects Git bytes and Python syntax, never private arrays or live jobs.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = 'research/wakeword_p4/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    repo = args.repository_root.resolve()
    ref = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], cwd=repo,
                                  text=True).strip()
    cache = {}

    def read(name):
        if name not in cache:
            cache[name] = subprocess.check_output(['git', 'show', ref + ':' + name], cwd=repo)
        return cache[name]

    def metadata(name):
        return json.loads(read(PREFIX + 'results/' + name))

    def source(name):
        return read(PREFIX + 'experiments/' + name)

    family = '4090-hey-broad-oneway-consistency-'
    formal = metadata(family + 'formal-execution-v1.json')
    smoke = metadata(family + 'smoke-execution-v1.json')
    parent = metadata('4090-hey-broad-paired-qat-formal-execution-v1.json')
    assert (formal['mode'], formal['steps'], formal['repeats']) == ('formal', 10000, 1)
    assert (smoke['mode'], smoke['steps'], smoke['repeats']) == ('smoke', 4, 2)
    for field in ['arms', 'optimizer', 'export', 'original_formal', 'versions']:
        assert formal[field] == smoke[field] == parent[field], field
    assert formal['conditions'] == smoke['conditions'] == ['twoway_consistency', 'oneway_consistency']
    assert {arm['target'] for arm in formal['arms']} == {'hey_willow', 'hey_nyx'}

    checked = {}
    for name, expected in formal['files'].items():
        normalized = name
        if '/' + PREFIX in name:
            normalized = PREFIX + name.split('/' + PREFIX, 1)[1]
        if normalized.startswith(PREFIX) and normalized.endswith(('.py', '.md', '.json')):
            actual = sha(read(normalized))
            assert actual == expected, normalized
            if normalized in checked:
                assert checked[normalized] == actual
            checked[normalized] = actual
    assert checked

    sources = [source('run_hey_broad_oneway_consistency_' + kind + '_v1.py')
               for kind in ['formal', 'smoke']]
    steps = []
    for data in sources:
        tree = ast.parse(data)
        step, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == 'step']
        steps.append(step)
    assert ast.dump(steps[0], include_attributes=False) == ast.dump(steps[1], include_attributes=False)
    # GPU uses Python 3.11; 3.12 added empty type_params to FunctionDef.
    # Remove only that empty field for the explicitly labelled 3.11 digest.
    for step in steps:
        for node in ast.walk(step):
            if 'type_params' in node._fields:
                assert node.type_params == []
                node._fields = tuple(field for field in node._fields if field != 'type_params')
    step_dump = ast.dump(steps[0], annotate_fields=True, include_attributes=False)
    derived = metadata(family + 'formal-derivation-v1.json')
    assert derived['full_update_step_ast_equal'] is True
    assert sha(step_dump.encode()) == derived['step_ast_sha256']
    for name, expected in derived['files'].items():
        assert sha(read(name)) == expected

    helper = ast.parse(source('hey_broad_oneway_consistency_loss_v1.py'))
    function, = [node for node in helper.body if isinstance(node, ast.FunctionDef)]
    assert function.name == 'components' and len(function.body) == 2
    imported, returned = function.body
    assert isinstance(imported, ast.ImportFrom) and imported.module == 'hey_broad_bounded_consistency_loss_v1'
    assert [(item.name, item.asname) for item in imported.names] == [('components', 'original')]
    expected = ast.parse('original(float_raw,quant_raw,tf.stop_gradient(float_bounded),'
                         'quant_bounded,labels,weights,coefficient,tf)', mode='eval').body
    assert isinstance(returned, ast.Return)
    assert ast.dump(returned.value, include_attributes=False) == ast.dump(expected, include_attributes=False)

    validation = metadata(family + 'validation-derivation-v1.json')
    assert len(validation['files']) == 7
    for item in validation['files']:
        before, after = read(item['original']), read(item['derived'])
        assert sha(before) == item['original_sha256'] and sha(after) == item['derived_sha256']
        assert before.replace(b'bounded_consistency', b'oneway_consistency').replace(
            b'bounded-consistency', b'oneway-consistency') == after
    continuation = metadata(family + 'continuation-plan-v1.json')
    assert all(continuation['files'][name] == value for name, value in formal['files'].items())
    assert continuation['pid'] > 0 and continuation['identity']['starttime'].isdigit()
    assert 'run_hey_broad_oneway_consistency_formal_v1.py run formal' in continuation['identity']['cmdline']

    guard = ast.parse(source('admit_hey_broad_oneway_consistency_numerical_v1.py'))
    gates_node, = [node.value for node in guard.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'GATES' for target in node.targets)]
    gates = {keyword.arg: ast.literal_eval(keyword.value) for keyword in gates_node.keywords}
    assert gates == metadata('4090-hey-broad-trained-quantization-plan-v1.json')['numerical_gates']
    smoke_result_bytes = read(PREFIX + 'results/' + family + 'smoke-result-v1.json')
    smoke_result = json.loads(smoke_result_bytes)
    smoke_audit = metadata(family + 'smoke-independent-audit-v1.json')
    assert smoke_result['all_passed'] is True and len(smoke_result['cases']) == 4
    assert smoke_audit['result_sha256'] == sha(smoke_result_bytes) and len(smoke_audit['cases']) == 4
    assert all(case['steps'] == 4 and case['fresh_update_repeats_exact'] is True
               for case in smoke_result['cases'])

    report = dict(
        schema='mac-oneway-formal-published-source-review-v1', gpu_ref=ref,
        script_sha256=sha(Path(__file__).read_bytes()),
        public_file_bindings_verified=len(checked),
        verified_public_file_bindings=checked,
        smoke_formal_complete_step_ast_exact=True,
        published_python311_step_ast_digest_reproduced=sha(step_dump.encode()),
        only_empty_python312_type_params_omitted=True,
        helper_only_stops_float_bounded_consistency_argument=True,
        seven_validation_sources_exact_namespace_derivatives=True,
        original_initialization_optimizer_schedule_export_versions_equal=True,
        formal_steps=10000, formal_repeats=1, smoke_steps=4, smoke_repeats=2,
        original_numerical_gates_exact=gates,
        continuation_identity=dict(pid=continuation['pid'], **continuation['identity']),
        continuation_inherits_all_formal_bindings=True,
        original_four_case_smoke_report_and_audit_chain_exact=True,
        all_source_checks_passed=True,
        limits='Published Git source/metadata only. Does not independently verify '
               'private batches, H5/Adam arrays, live GPU process, gradient execution, '
               'training completion or model quality. Formal repeats=1 provides no '
               'full10000 fresh-repeat proof. Smoke repeats=2 covers only4 updates. '
               'No inference, new scoring, fresh development or acceptance access. '
               'All previous rejections and original gates remain.')
    with args.report.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(all_source_checks_passed=True,
                          public_file_bindings=len(checked), validation_derivatives=7,
                          report_sha256=sha(args.report.read_bytes()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
