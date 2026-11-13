"""Fresh development-only complete decode/frontend; reuse pinned original pipeline."""
import argparse
import ast
import json
from pathlib import Path
import prepare_hey_earnings_development_v1 as original

RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-jenny-earnings-preparation-plan-v1.json'
RESERVATION=RESULTS/'4090-jenny-earnings-development-reservation-v1.json'
OUT=Path('.local/wakeword/jenny-earnings-development-v1')
ORIGINAL=Path(original.__file__)


def environment():
    old=json.loads((RESULTS/'4090-hey-earnings-development-preparation-plan-v1.json').read_text())
    expected=next(h for p,h in old['files'].items() if p.endswith('/prepare_hey_earnings_development_v1.py'))
    assert original.sha(ORIGINAL)==expected
    tree=ast.parse(ORIGINAL.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    planfn=next(n for n in funcs if n.name=='make_plan')
    assert ast.unparse(planfn.body[0]).startswith('files = ')
    planfn.body[1:1]=ast.parse('files.append(ORIGINAL)').body
    class Identity(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='hey-earnings-development-preparation-plan-v1':
                return ast.copy_location(ast.Constant('jenny-earnings-development-preparation-plan-v1'),n)
            return n
    env=vars(original).copy()
    env.update(__file__=__file__,ORIGINAL=ORIGINAL,PLAN=PLAN,RESERVATION=RESERVATION,OUT=OUT)
    exec(compile(ast.fix_missing_locations(Identity().visit(ast.Module(body=funcs,type_ignores=[]))),str(ORIGINAL),'exec'),env)
    return env


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','prepare']);a=p.parse_args()
    env=environment()
    if a.action=='plan':env['make_plan']()
    else:
        env['prepare']()
        m=json.loads((OUT/'manifest.json').read_text())
        assert m['all_complete'] and m['integrity_preparation_passed'] and len(m['records'])==12
        report=dict(schema='jenny-earnings-complete-preparation-v1',plan_sha256=original.sha(PLAN),
            manifest_path=str(OUT/'manifest.json'),manifest_sha256=original.sha(OUT/'manifest.json'),
            transcript_audit_sha256=original.sha(OUT/'transcript-audit.json'),hours=m['hours'],sources=12,
            overlap_admitted=False,scoring_authorized=False,acceptance_accessed=False,
            scope='Original full decode/one-frontend-per-call pipeline, separate fresh development sources/output. Exact history audit required before scoring.')
        with (RESULTS/'4090-jenny-earnings-complete-preparation-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
