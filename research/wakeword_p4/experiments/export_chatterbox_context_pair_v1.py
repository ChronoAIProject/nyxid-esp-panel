"""One final INT8 export per paired training arm; no checkpoint/calibration search."""
import ast
import json
from pathlib import Path
import sys
import export_c161_chatterbox_v1 as exercised
from train_chatterbox_context_pair_v1 import lock as training_lock,output as training_output,report as training_report
R=exercised.R

def lock(condition):return R/f'4090-context-{condition}-int8-plan-v1.json'
def output(condition):return Path(f'.local/wakeword/context-{condition}-int8-v1')
def report(condition):return R/f'4090-context-{condition}-int8-original-v1.json'

def plan(condition):
    formal=training_report(condition)
    if not formal.exists():
        print(json.dumps(dict(status='not_ready',condition=condition,missing=str(formal),model_calls=0)));return 2
    r=json.loads(formal.read_text());parent=training_lock(condition);p=json.loads(parent.read_text())
    assert r['plan_sha256']==exercised.sha(parent) and r['all_training_checks_passed']
    assert r['steps']==r['optimizer_iterations']==20000
    weights=training_output(condition)/'final.weights.h5'
    assert exercised.sha(weights)==r['files']['final.weights.h5']
    representative=p['export_representative'];assert representative['sentinels']==[0,26] and representative['intermediate_reset'] is False
    paths=[Path(__file__),Path(exercised.__file__),exercised.SOURCE,formal,parent,weights,R/'4090-causal-sequence-smoke-int8-original-v1.json',Path(representative['batch'])]
    files=dict(p['files']);files.update({str(f):exercised.sha(f) for f in paths})
    record=dict(schema=f'context-{condition}-int8-plan-v1',condition=condition,files=files,weights=str(weights),batch=representative['batch'],lengths=representative['lengths'],versions={n:exercised.common.importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},expected_new_gib=.1,retries=0,policy='One fixedfinal20000 INT8 export for this paired arm, same exercised builtinINT8 ABI/context309/realzero states/rawsigmoid. Retain old calibration representative batch unchanged as control; not actual first batch of this arm. Zero/26 sentinels then16saved sequences, no intermediate reset. Static ABI and one fresh-interpreter functional replay only, no FP32 equivalence or calibration retry. No checkpoint/threshold rescue, acceptance or qualification. Both arm outcomes reported, C161 retained.')
    with lock(condition).open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='planned',condition=condition,plan_sha256=exercised.sha(lock(condition)))));return 0

def run(condition):
    # Reuse the exact historical wrapper, substituting only output identity/schema.
    fn=next(n for n in ast.parse(Path(exercised.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='c161-chatterbox-int8-original-v1':return ast.copy_location(ast.Constant(f'context-{condition}-int8-original-v1'),n)
            return n
    env=dict(vars(exercised));env.update(LOCK=lock(condition),OUT=output(condition),REPORT=report(condition))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()

if __name__=='__main__':
    action,condition=sys.argv[1:];assert condition in ['silence','speech'] and action in ['plan','run']
    raise SystemExit(plan(condition) if action=='plan' else run(condition))
