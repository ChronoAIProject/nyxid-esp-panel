"""Existing Joe development only; native full-utterance boundaries, no final access."""
import argparse,ast,json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R;LOCK=R/'4090-causal-source-balanced-joe-plan-v1.json';OUT=Path('.local/wakeword/causal-source-balanced-joe-v1')
def plan():
    parent=R/'4090-causal-joe-development-plan-v1.json';p=json.loads(parent.read_text())
    original=R/'4090-causal-source-balanced-int8-original-v1.json';report=json.loads(original.read_text());assert report['all_static_and_runtime_abi_passed'];m=report['model'];assert common.sha(m['path'])==m['sha256']
    files={f:h for f,h in p['files'].items() if not f.endswith('.tflite')}
    for f in [parent,original,Path(m['path']),Path(__file__),Path(__file__).with_name('score_causal_source_balanced_joe_v1.py')]:files[str(f)]=common.sha(f)
    p.update(schema='causal-source-balanced-joe-plan-v1',files=files,models=[dict(target='hey_nyx',condition='source-balanced',path=m['path'],sha256=m['sha256'])],policy='One new source-balanced candidate on existing consumedJoe360wholewaves, fixed.91796875/smoothing5/continuous81600prime. Prior primedv30/9307/95ab traces retained, no repeated baseline inference. Joe is development and never added to training; no protectedMike/finaldata access. No synthesis/render/tuning. Only generatedwholecontainer boundaries, no invented wakeprefix. No backgroundFAR claim, independent acceptance or promotion.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(common.sha(LOCK))
def run():
    fn=next(n for n in ast.parse(Path(common.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==372:return ast.copy_location(ast.Constant(360),n)
            if n.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-source-balanced-joe-original-v1'),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='completed' for k in n.keywords):
                for k in n.keywords:
                    if k.arg=='models':k.value=ast.Constant(1)
                    if k.arg=='traces':k.value=ast.Constant(360)
            return n
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT);exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
