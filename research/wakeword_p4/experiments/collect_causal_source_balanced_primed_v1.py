"""Actual existing primed deployment path; cold is an initialization comparison."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R
LOCK=R/'4090-causal-source-balanced-primed-plan-v1.json'
OUT=Path('.local/wakeword/causal-source-balanced-primed-v1')
PRIME=common.PRIME
SOURCE=Path(common.__file__)
sha=common.sha

def plan():
    parent=R/'4090-causal-sequence-formal-primed-plan-v1.json';p=json.loads(parent.read_text())
    original=R/'4090-causal-source-balanced-int8-original-v1.json';r=json.loads(original.read_text());assert r['all_static_and_runtime_abi_passed']
    model=r['model'];assert sha(model['path'])==model['sha256']
    files=dict(p['files']);paths=[Path(__file__),SOURCE,parent,original,Path(model['path']),Path(__file__).with_name('score_causal_source_balanced_primed_v1.py')]
    files.update({str(path):sha(path) for path in paths})
    lock=dict(schema='causal-source-balanced-primed-plan-v1',files=files,versions=p['versions'],models=[dict(target='hey_nyx',condition='source-balanced',path=model['path'],sha256=model['sha256'])],materials=p['materials'],grid=[.91796875],prime_samples_requested=81440,prime_samples_actual=PRIME,expected_new_gib=.3,retries=0,reference_threshold=.91796875,baseline_plan_sha256=sha(parent),baseline_result_sha256=sha(Path('.local/wakeword/causal-sequence-formal-primed-v1/result.json')),policy='One new source-balanced candidate on the identical consumed372 fullwaveforms/81600continuousprime, no transition reset. Retain priming events/state, all firstphrases/full10.483944826388889h. Reference .91796875 fixed;One fixed reference point for actual behavior, not thresholdselection. Compare retained completed prior causal/v30 on identical protocol without rerunning them. Commands measured by generated-prefix and full-container clocks, not unknown phonetic truth. No numericalalignment audit/newTTS/freshroles/qualification; v30 retained pending all independent requirements.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),models=1,inputs=372,prime_samples=PRIME)))

def run():
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='completed' for k in node.keywords):
                for k in node.keywords:
                    if k.arg in ('models','traces'):k.value=ast.Constant(1 if k.arg=='models' else 372)
            return node
        def visit_Constant(self,node):
            if node.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-source-balanced-primed-original-v1'),node)
            return node
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)],type_ignores=[])),str(SOURCE),'exec'),env)
    env['run']()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
