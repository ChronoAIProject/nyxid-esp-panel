"""Reuse authenticated independent sparse-RIR replay, no model calls."""
import ast
import json
from pathlib import Path

import audit_hey_broad_render_v1 as original


def main():
    source = Path(original.__file__)
    results = Path('research/wakeword_p4/results')
    receipt = json.loads((results / '4090-hey-broad-render-independent-audit-v1.json').read_text())
    assert original.sha(source) == receipt['script_sha256']
    plan = json.loads((results / '4090-multispeaker-coverage-render-plan-v1.json').read_text())
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    replacements = {
        '4090-hey-broad-render-plan-v1.json': '4090-multispeaker-coverage-render-plan-v1.json',
        '4090-hey-broad-render-v1.json': '4090-multispeaker-coverage-render-v1.json',
        'hey-broad-render-independent-audit-v1': 'multispeaker-coverage-render-independent-audit-v1',
        '4090-hey-broad-render-independent-audit-v1.json': '4090-multispeaker-coverage-render-independent-audit-v1.json',
        'context225_full_source_prefix_failures': 'context_full_source_container_failures',
        106956: 23220, 48000: plan['lead_samples'], 225: plan['context_frames'],
    }

    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value in replacements:
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            if isinstance(node.value, str) and 'context225 maxjitter27' in node.value:
                return ast.copy_location(ast.Constant(node.value.replace(
                    'context225 maxjitter27', f"context{plan['context_frames']} whole-container maxjitter27")), node)
            return node

    env = vars(original).copy()
    env.update(BASE=Path('.local/wakeword/multispeaker-coverage-rendered-v1'), __file__=__file__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)], type_ignores=[])), str(source), 'exec'), env)
    env['main']()


if __name__ == '__main__':
    main()
