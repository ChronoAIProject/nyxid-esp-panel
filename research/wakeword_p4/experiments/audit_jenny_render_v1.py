"""Reuse independent sparse4tap replay on new development materials only."""
import ast
import json
from pathlib import Path
import audit_hey_broad_render_v1 as original


def main():
    source=Path(original.__file__)
    old=json.loads(Path('research/wakeword_p4/results/4090-hey-broad-render-independent-audit-v1.json').read_text())
    assert original.sha(source)==old['script_sha256']
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    replacements={
        '4090-hey-broad-render-plan-v1.json':'4090-jenny-development-render-plan-v1.json',
        '4090-hey-broad-render-v1.json':'4090-jenny-development-render-v1.json',
        'hey-broad-render-independent-audit-v1':'jenny-development-render-independent-audit-v1',
        '4090-hey-broad-render-independent-audit-v1.json':'4090-jenny-development-render-independent-audit-v1.json',
        106956:396}
    class Rename(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    env=vars(original).copy();env.update(BASE=Path('.local/wakeword/jenny-development-rendered-v1'),__file__=__file__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Rename().visit(fn)],type_ignores=[])),str(source),'exec'),env)
    env['main']()


if __name__=='__main__':main()
