"""One bc01 replay of consumed Melo474; fixed threshold, no grid selection."""
import ast
import json
from pathlib import Path
import sys
import collect_melo_development_v2 as parent

R = parent.R
LOCK = R / '4090-bc01-melo-plan-v1.json'
OUT = Path('.local/wakeword/bc01-melo-v1')


def plan():
    previous = R / '4090-melo-development-int8-execution-v2.json'
    p = json.loads(previous.read_text())
    export = R / '4090-c161-chatterbox-int8-original-v1.json'
    e = json.loads(export.read_text())
    assert e['all_static_and_runtime_abi_passed']
    model = e['model']
    assert parent.sha(model['path']) == model['sha256']
    p['files'].update({str(path): parent.sha(path) for path in
        [Path(__file__), Path(parent.__file__), previous, export, Path(model['path'])]})
    p.update(schema='bc01-consumed-melo-plan-v1',
        models=[dict(id='bc01', path=model['path'], sha256=model['sha256'], role='experimental-development')],
        reference_threshold=.91796875, expected_new_gib=.1,
        policy='All474 previously consumed Melo recordings:432synthetic+42complete background. Existing source rows retain historical original role labels for identity only; not fresh threshold-development. Fresh bc01 frontend/interpreter per full recording, continuous81600zero prime/no boundary reset/native480 clock. Compare retained C161 traces at .91796875/5 only, no threshold grid or selection. EN-BR means British. Background reused not new/additive hours. No training or acceptance access, no P4/human qualification.')
    with LOCK.open('x') as stream:
        stream.write(json.dumps(p, indent=2) + '\n')


def run():
    function = next(n for n in ast.parse(Path(parent.__file__).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run')
    class Adapter(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'dict':
                for keyword in node.keywords:
                    if keyword.arg == 'models' and isinstance(keyword.value, ast.Constant):
                        keyword.value = ast.Constant(1)
            return node
    env = dict(vars(parent), LOCK=LOCK, OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)], type_ignores=[])),
                 str(__file__), 'exec'), env)
    env['run']()


if __name__ == '__main__':
    {'plan': plan, 'run': run}[sys.argv[1]]()
