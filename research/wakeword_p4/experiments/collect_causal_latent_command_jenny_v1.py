"""One new latent-command candidate on consumed Jenny development; no final access."""
import argparse,ast,json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R
LOCK=R/'4090-causal-latent-command-jenny-plan-v1.json'
OUT=Path('.local/wakeword/causal-latent-command-jenny-v1')
def plan():
    parent=R/'4090-causal-nearword-jenny-plan-v1.json';p=json.loads(parent.read_text())
    original=R/'4090-causal-latent-command-int8-original-v1.json';r=json.loads(original.read_text());assert r['all_static_and_runtime_abi_passed'];m=r['model'];assert common.sha(m['path'])==m['sha256']
    files={f:h for f,h in p['files'].items() if not f.endswith('.tflite')}
    for f in [parent,original,Path(m['path']),Path(__file__),Path(__file__).with_name('score_causal_latent_command_jenny_v1.py')]:files[str(f)]=common.sha(f)
    p.update(schema='causal-latent-command-jenny-plan-v1',files=files,models=[dict(target='hey_nyx',condition='latent-command',path=m['path'],sha256=m['sha256'])],policy='One new original latent-command INT8 model on consumed Jenny396 plus complete9.614h earnings12, fixed .91796875/smoothing5 and actual continuous81600prime. No fitting, threshold search, newTTS, waveform crop or final access. Prior candidate/v30 raw evidence reused without repeated inference. Development only, not independent acceptance; generated-boundary timing is not phonetic truth.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(common.sha(LOCK))
def run():
    fn=next(n for n in ast.parse(Path(common.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==372:return ast.copy_location(ast.Constant(408),n)
            if n.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-latent-command-jenny-original-v1'),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='completed' for k in n.keywords):
                for k in n.keywords:
                    if k.arg=='models':k.value=ast.Constant(1)
                    if k.arg=='traces':k.value=ast.Constant(408)
            return n
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT);exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
