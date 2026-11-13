"""Fresh12 background exact-history adapter; no old media/model inference."""
import ast
import json
from pathlib import Path
import audit_hey_earnings_history_overlap_v1 as original


def main():
    source=Path(original.__file__)
    old=json.loads(Path('research/wakeword_p4/results/4090-hey-earnings-history-overlap-v1.json').read_text())
    assert original.sha(source)==old['script_sha256']
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    mapping={'research/wakeword_p4/results/4090-hey-earnings-development-preparation-plan-v1.json':'research/wakeword_p4/results/4090-jenny-earnings-preparation-plan-v1.json',
        'research/wakeword_p4/results/4090-hey-earnings-background-reservation-v1.json':'research/wakeword_p4/results/4090-jenny-earnings-development-reservation-v1.json',
        'hey-earnings-history-overlap-audit-v1':'jenny-earnings-history-overlap-audit-v1'}
    class Rewrite(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value in mapping:return ast.copy_location(ast.Constant(mapping[n.value]),n)
            return n
    base=Path('.local/wakeword');out=base/'jenny-earnings-history-audit-v1'
    env=vars(original).copy()
    env.update(__file__=__file__,PREP=base/'jenny-earnings-development-v1',OUT=out,
        RESULT=Path('research/wakeword_p4/results/4090-jenny-earnings-history-overlap-v1.json'),
        COMMITS=['87ac9469','d0c589b8357b3b41127a724ed5f63df64a68d3b7'],
        EXCLUDED=[out]+[p for p in base.iterdir() if p.name.startswith('jenny-')])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Rewrite().visit(fn)],type_ignores=[])),str(source),'exec'),env)
    env['main']()


if __name__=='__main__':main()
