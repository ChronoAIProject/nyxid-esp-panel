"""One fixedfinal20000 INT8 export, sharing the already exercised ABI exporter."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import export_causal_sequence_smoke_v1 as common
R=common.R
LOCK=R/'4090-c161-chatterbox-int8-plan-v1.json'
OUT=Path('.local/wakeword/c161-chatterbox-int8-v1')
REPORT=R/'4090-c161-chatterbox-int8-original-v1.json'
SOURCE=Path(common.__file__)
sha=common.sha

def plan():
    formal=R/'4090-c161-chatterbox-training-original-v1.json'
    if not formal.exists():
        print(json.dumps(dict(status='not_ready',missing=str(formal),model_calls=0)));return 2
    r=json.loads(formal.read_text());parent=R/'4090-c161-chatterbox-training-plan-v1.json';p=json.loads(parent.read_text())
    assert r['plan_sha256']==sha(parent) and r['all_training_checks_passed'] and r['steps']==r['optimizer_iterations']==20000
    weights=Path('.local/wakeword/c161-chatterbox-training-v1/final.weights.h5');assert sha(weights)==r['files']['final.weights.h5']
    representative=p['export_representative'];assert representative['sentinels']==[0,26] and representative['intermediate_reset'] is False
    files=dict(p['files']);paths=[Path(__file__),SOURCE,formal,parent,weights,R/'4090-causal-sequence-smoke-int8-original-v1.json']
    files.update({str(x):sha(x) for x in paths})
    lock=dict(schema='c161-chatterbox-int8-plan-v1',files=files,weights=str(weights),batch=representative['batch'],lengths=representative['lengths'],versions={n:common.importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},expected_new_gib=.1,retries=0,policy='Original fixedfinal20000 causal candidate only. Conditioned Chatterbox training-only source expansion; same exercised one builtinINT8 exporter/context309/sixrealzero states/frozenBN/rawsigmoid. Representative material fixed before formal training: zero/26 then first16actualbatch1/actual lengths/final<=2featurezero pads; no intermediate reset. StaticABI and one fresh interpreter functional replay, no FP32 export/comparison or calibration retry. Basic file integrity/interface/reset/function checks only, no savedstate numericalaudit or equality gate. Actual complete waveform INT8 recognition follows. No threshold selection/freshdev/acceptance/qualification here; v30 retained.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),steps=20000,weights_sha256=sha(weights))))
    return 0

def run():
    # Share the exact previously exercised conversion/ABI implementation without
    # editing its historical execution-bound source or disposable report bytes.
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='before' for t in node.targets):return None
            return self.generic_visit(node)
        def visit_Assert(self,node):
            source=ast.unparse(node.test)
            if 'np.array_equal(*traces)' in source:
                return ast.copy_location(ast.Assert(test=ast.parse('len(traces[0])==len(sequences[0])//3',mode='eval').body,msg=None),node)
            if 'zip(model.variables, before' in source:
                return ast.copy_location(ast.Assert(test=ast.parse("sha(p['weights'])==p['files'][p['weights']]",mode='eval').body,msg=None),node)
            return self.generic_visit(node)
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==2:node.args[0]=ast.Constant(1)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='savez_compressed':node.keywords=[k for k in node.keywords if k.arg!='second']
            if isinstance(node.func,ast.Name) and node.func.id=='dict':
                for k in node.keywords:
                    if k.arg=='repeats_exact':k.arg='repeat_comparison_performed';k.value=ast.Constant(False)
                    if k.arg=='calls_per_repeat':k.arg='functional_replay_calls'
                    if k.arg=='source_weights_unchanged':k.arg='source_weights_file_unchanged'
            return node
        def visit_Constant(self,node):
            substitutions={'4090-causal-sequence-smoke-int8-original-v1.json':REPORT.name,'causal-sequence-smoke-int8-original-v1':'c161-chatterbox-int8-original-v1'}
            if isinstance(node.value,str) and node.value in substitutions:return ast.copy_location(ast.Constant(substitutions[node.value]),node)
            return node
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)],type_ignores=[])),str(SOURCE),'exec'),env)
    env['run']()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();raise SystemExit(plan() if a.action=='plan' else run())
