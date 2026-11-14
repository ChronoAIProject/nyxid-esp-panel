"""Exercise exact receipt reduction code on original saved270 arrays, no inference."""
import ast
import copy
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha

ROOT=Path('research/wakeword_p4')
SOURCE=ROOT/'experiments/receipt_hey_broad_bounded_v1.py'
OUT=ROOT/'results/4090-bounded-receipt-array-audit-review-v1.json'

def main():
    assert not OUT.exists()
    tree=ast.parse(SOURCE.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.iter)=="screen['cases']")
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(SOURCE),'exec')
    original=ROOT/'results/4090-hey-broad-trained-quantization-result-v1.json'
    plan=ROOT/'results/4090-hey-broad-trained-quantization-plan-v1.json'
    screen=json.loads(original.read_text());gates=json.loads(plan.read_text())['numerical_gates']
    env=dict(np=np,sha=sha,screen=screen,gates=gates,audited=[])
    exec(code,env)
    audited=env['audited'];assert len(audited)==270
    summary={t:sum(r['passed'] for r in audited if r['target']==t) for t in ['hey_willow','hey_nyx']}
    assert summary==dict(hey_willow=54,hey_nyx=117)
    checks={}
    for name,mutate in [('false_pass',lambda r:r.update(passed=not r['passed'])),('altered_statistic',lambda r:r['smoothed_abs_error'].update(mean=r['smoothed_abs_error']['mean']+1))]:
        row=copy.deepcopy(screen['cases'][0]);mutate(row)
        env=dict(np=np,sha=sha,screen=dict(cases=[row]),gates=gates,audited=[])
        try:exec(code,env)
        except AssertionError:checks[name]=True
        else:raise AssertionError(name+' was not rejected')
    result=dict(schema='bounded-receipt-array-audit-review-v1',source_sha256=sha(SOURCE),original_result_sha256=sha(original),review_sha256=sha(__file__),original_arrays_verified=270,original_gate_pass_counts=summary,tampering_rejected=checks,limits='Exact receipt loop extracted by AST and exercised on original consumed saved training arrays. No model/native frontend invoke, conversion, new data, candidate pass, or actual bounded receipt. Static state audit and complete receipt main await real bounded export/screen artifacts.')
    with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
