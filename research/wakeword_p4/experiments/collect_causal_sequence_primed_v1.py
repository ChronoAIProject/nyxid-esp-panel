"""Actual existing primed deployment path; cold is an initialization comparison."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R
LOCK=R/'4090-causal-sequence-formal-primed-plan-v1.json'
OUT=Path('.local/wakeword/causal-sequence-formal-primed-v1')
PRIME=common.PRIME
SOURCE=Path(common.__file__)
sha=common.sha

def plan():
    parent=R/'4090-causal-sequence-formal-development-plan-v1.json';p=json.loads(parent.read_text())
    original=R/'4090-causal-sequence-formal-int8-original-v1.json';r=json.loads(original.read_text());assert r['all_static_and_runtime_abi_passed']
    files=dict(p['files']);paths=[Path(__file__),SOURCE,parent,original,Path(__file__).with_name('score_causal_sequence_primed_v1.py'),R/'4090-causal-sequence-formal-development-exit-v1.json',R/'4090-causal-sequence-final-int8-mac-primed-handoff-v1.json']
    files.update({str(path):sha(path) for path in paths})
    for path,h in files.items():assert sha(path)==h,path
    assert PRIME==81600 and len(p['models'])==2 and len(p['grid'])==138
    lock=dict(schema='causal-sequence-existing-primed-primary-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},models=p['models'],materials=p['materials'],grid=p['grid'],prime_samples_requested=81440,prime_samples_actual=PRIME,expected_new_gib=.5,retries=0,mac_revision_comments=[6885081704,6885130975,6885267787],policy='Existing deployment primary:81600zeroPCM then complete unchanged original PCM, one continuous frontend/model/threeframe/event/arming/cooldown state. Not a new guard/readiness wait. Preserve allprimecodes/events, ignore only app notifications during prime, keep every real firstphrase and original full10.483944826388889h background/model. Same causal60952B and original frozenv3054440B, no re-export/weight change. All138secondaryUINT8points each plus separatev30 originalfrozenfloat64convolution point. Original complete final160hop/featurezero finish only at end, never priming boundary. Cold originals remain initialization comparisons and cannot be an extra veto for this deployment. No newthreshold/waitselection, freshdev/acceptance, audio synthesis or numericalalignment gate. Candidate remains unqualified pending actual primary behavior and independent requirements; do not infer rejection solely from cold.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),models=2,inputs=372,prime_samples=PRIME)))

def run():
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-sequence-existing-primed-primary-original-v1'),node)
            return node
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)],type_ignores=[])),str(SOURCE),'exec'),env)
    env['run']()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
