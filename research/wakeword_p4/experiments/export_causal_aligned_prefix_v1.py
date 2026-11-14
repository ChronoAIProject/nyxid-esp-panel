"""One fixedfinal20000 INT8 export, sharing the already exercised ABI exporter."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import export_causal_sequence_smoke_v1 as common
R=common.R
LOCK=R/'4090-causal-aligned-prefix-int8-plan-v1.json'
OUT=Path('.local/wakeword/causal-aligned-prefix-int8-v1')
REPORT=R/'4090-causal-aligned-prefix-int8-original-v1.json'
SOURCE=Path(common.__file__)
sha=common.sha

def plan():
    formal=R/'4090-causal-aligned-prefix-original-v1.json'
    if not formal.exists():
        print(json.dumps(dict(status='not_ready',missing=str(formal),model_calls=0)));return 2
    r=json.loads(formal.read_text());parent=R/'4090-causal-aligned-prefix-plan-v1.json';p=json.loads(parent.read_text())
    assert r['plan_sha256']==sha(parent) and r['all_training_checks_passed'] and r['steps']==r['optimizer_iterations']==20000
    weights=Path('.local/wakeword/causal-aligned-prefix-v1/final.weights.h5');assert sha(weights)==r['files']['final.weights.h5']
    # Only saved initialization/frozen-state integrity; no batch reconstruction,
    # score agreement, gradient comparison or checkpoint selection is required.
    root=weights.parent
    with np.load(root/'initial-model-variables.npz') as initial,np.load(root/'final-model-variables.npz') as final:
        assert len(initial.files)==len(final.files)==31
        for row in p['initial']['variable_inventory']['variables']:
            a=initial[row['key']];b=final[row['key']]
            assert common.hashlib.sha256(a.tobytes()).hexdigest()==row['sha256']
            assert np.isfinite(b).all()
            if not row['effective_trainable']:assert np.array_equal(a,b)
    representative=p['export_representative'];assert representative['sentinels']==[0,26] and representative['intermediate_reset'] is False
    files=dict(p['files']);paths=[Path(__file__),SOURCE,formal,parent,weights,root/'initial-model-variables.npz',root/'final-model-variables.npz',R/'4090-causal-sequence-smoke-int8-original-v1.json']
    files.update({str(x):sha(x) for x in paths})
    lock=dict(schema='causal-aligned-prefix-int8-plan-v1',files=files,weights=str(weights),batch=representative['batch'],lengths=representative['lengths'],versions={n:common.importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},expected_new_gib=.1,retries=0,policy='Original fixedfinal20000 causal candidate only. Aligned-prefix correction; same exercised one builtinINT8 exporter/context309/sixrealzero states/frozenBN/rawsigmoid. Representative material fixed before formal training: zero/26 then first16actualbatch1/actual lengths/final<=2featurezero pads; no intermediate reset. StaticABI/fresh interpreter repeat, no FP32 export/comparison or calibration retry. Savedinitial/frozenstate integrity only, no fullbatchaudit/numericalagreement prerequisite. Actual complete waveform INT8 recognition follows. No threshold selection/freshdev/acceptance/qualification here; v30 retained.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),steps=20000,weights_sha256=sha(weights))))
    return 0

def run():
    # Share the exact previously exercised conversion/ABI implementation without
    # editing its historical execution-bound source or disposable report bytes.
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,node):
            substitutions={'4090-causal-sequence-smoke-int8-original-v1.json':REPORT.name,'causal-sequence-smoke-int8-original-v1':'causal-aligned-prefix-int8-original-v1'}
            if isinstance(node.value,str) and node.value in substitutions:return ast.copy_location(ast.Constant(substitutions[node.value]),node)
            return node
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)],type_ignores=[])),str(SOURCE),'exec'),env)
    env['run']()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();raise SystemExit(plan() if a.action=='plan' else run())
