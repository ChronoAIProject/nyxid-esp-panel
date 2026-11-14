"""Review published formal/continuation source identities without GPU access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = 'research/wakeword_p4/'


def run(args):
    ref = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    cache = {}

    def read(name):
        if name not in cache:
            cache[name] = subprocess.check_output(['git', 'show', ref+':'+name])
        return cache[name]

    def metadata(name):
        return json.loads(read(PREFIX+'results/'+name))

    formal = metadata('4090-hey-broad-bounded-consistency-formal-execution-v1.json')
    parent = metadata('4090-hey-broad-paired-qat-formal-execution-v1.json')
    smoke = metadata('4090-hey-broad-bounded-consistency-smoke-execution-v1.json')
    assert formal['mode'] == 'formal' and formal['steps'] == 10000 and formal['repeats'] == 1
    assert smoke['mode'] == 'smoke' and smoke['steps'] == 4 and smoke['repeats'] == 2
    assert formal['arms'] == parent['arms'] == smoke['arms']
    assert formal['optimizer'] == parent['optimizer'] == smoke['optimizer']
    assert formal['export'] == parent['export'] == smoke['export']
    assert formal['original_formal'] == parent['original_formal'] == smoke['original_formal']
    assert formal['conditions'] == smoke['conditions'] == ['raw_logit_control', 'bounded_logit_consistency']
    checked = {}
    for name, expected in formal['files'].items():
        if name.startswith(PREFIX) and name.endswith(('.py', '.md', '.json')):
            actual = hashlib.sha256(read(name)).hexdigest()
            assert actual == expected, name
            checked[name] = actual
    sources = [read(PREFIX+'experiments/'+name) for name in
               ['run_hey_broad_bounded_consistency_formal_v1.py', 'run_hey_broad_bounded_consistency_v1.py']]
    trees = [ast.parse(source) for source in sources]
    steps = [[node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == 'step'] for tree in trees]
    assert all(len(found) == 1 for found in steps)
    dumps = [ast.dump(found[0], include_attributes=False) for found in steps]
    assert dumps[0] == dumps[1]
    derived = metadata('4090-hey-broad-bounded-consistency-formal-derivation-v1.json')
    # The published digest has no serialization label. Independently compare
    # complete step ASTs and record our explicit encoding, rather than assume it.
    independent_ast_digest = hashlib.sha256(dumps[0].encode()).hexdigest()
    assert [hashlib.sha256(source).hexdigest() for source in sources] == [
        derived['formal_source_sha256'], derived['smoke_source_sha256']]
    options = [node for node in ast.walk(trees[0]) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_experimental_options']
    assert len(options) == 1 and ast.literal_eval(options[0].args[0]) == {
        'constant_folding': False, 'arithmetic_optimization': False}
    legacy_field = [node for node in ast.walk(trees[0]) if isinstance(node, ast.keyword)
                    and node.arg == 'fresh_update_repeats_exact']
    assert len(legacy_field) == 1 and ast.literal_eval(legacy_field[0].value) is True
    continuation = metadata('4090-hey-broad-bounded-consistency-continuation-plan-v1.json')
    launch = metadata('4090-hey-broad-bounded-consistency-formal-launch-v1.json')
    assert continuation['pid'] == launch['pid']
    assert continuation['identity']['starttime'] == launch['starttime']
    assert continuation['identity']['cmdline'].strip() == launch['cmdline'].strip()
    assert all(continuation['files'][name] == value for name, value in formal['files'].items())
    guard = ast.parse(read(PREFIX+'experiments/admit_hey_broad_bounded_consistency_numerical_v1.py'))
    gate_node, = [node.value for node in guard.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'GATES' for target in node.targets)]
    gates = {keyword.arg: ast.literal_eval(keyword.value) for keyword in gate_node.keywords}
    assert gates == metadata('4090-hey-broad-trained-quantization-plan-v1.json')['numerical_gates']
    result = dict(schema='mac-bounded-formal-published-source-review-v1', gpu_ref=ref,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        published_formal_source_bindings_verified=checked,
        smoke_formal_update_step_ast_exact=True,
        independent_ast_dump_include_attributes_false_sha256=independent_ast_digest,
        gpu_declared_ast_sha256=derived['update_step_ast_sha256'],
        declared_ast_digest_serialization_independently_verified=False,
        original_initialization_optimizer_export_schedule_bindings_match=True,
        conditions=formal['conditions'], formal_steps=10000, formal_repeats=1,
        smoke_steps=4, smoke_repeats=2, original_numerical_gates_exact=gates,
        continuation_binds_original_launch_identity=True,
        declared_ast_digest_limit='Published derivation omits digest serialization. Source bytes and direct full AST equality independently verified; do not claim reproduction of its unlabeled AST digest.',
        metadata_scope_finding='Formal source still emits fresh_update_repeats_exact=True, inherited from smoke, but formal repeats=1 makes that check vacuous. It provides no full10000 fresh model/optimizer repeat proof. Preserve running locked source/original report; add explicit single-run scope clarification in an additive receipt, do not restart or require another training run.',
        all_source_checks_passed=True,
        limits='Published source/metadata identity review only. No private H5/batches/actual GPU process or complete outputs read. Does not prove training completion, model/optimizer state, full-run repeat, numerical quality, recognition/FAR or development admission. The scope finding is reporting accuracy, not a training failure.')
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(args.report), all_source_checks_passed=True,
                         source_bindings=len(checked), formal_repeats=1,
                         inherited_repeat_field_does_not_prove_fullrun_repetition=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
