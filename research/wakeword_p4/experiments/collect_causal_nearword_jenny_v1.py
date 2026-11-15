"""Fixed-profile practical replay of existing Jenny development materials only."""
import argparse,ast,json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R
LOCK=R/'4090-causal-nearword-jenny-plan-v1.json'
OUT=Path('.local/wakeword/causal-nearword-jenny-v1')
def plan():
    materials={}
    files={}
    for phase,receipt,key in [('synthetic','4090-jenny-development-replay-manifest-v1.json','path'),('background','4090-jenny-earnings-complete-preparation-v1.json','manifest_path')]:
        rp=R/receipt;r=json.loads(rp.read_text());p=Path(r[key]);assert common.sha(p)==r['sha256' if phase=='synthetic' else 'manifest_sha256']
        materials[phase]=dict(path=str(p),sha256=common.sha(p),role='existing development; not final acceptance')
        files[str(rp)]=common.sha(rp);files[str(p)]=common.sha(p)
        rows=json.loads(p.read_text())['records'];assert len(rows)==(396 if phase=='synthetic' else 12)
        for row in rows:
            f=Path(row['path'] if phase=='synthetic' else row['wav_path']);files[str(f)]=common.sha(f)
    models=[]
    for condition,path,expected in [('nearword','.local/wakeword/causal-existing-nearword-int8-v1/int8.tflite','9307e6c13b7bc269f2566f111acdddcd8a2fedac569ce9f5c35b4e9fd030a7d6'),('v30','.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite','04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c')]:
        assert common.sha(path)==expected;files[path]=expected;models.append(dict(target='hey_nyx',condition=condition,path=path,sha256=expected))
    for f in [Path(__file__),Path(common.__file__),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path('research/wakeword_p4/training/features.py')]:files[str(f)]=common.sha(f)
    p=dict(schema='causal-nearword-jenny-plan-v1',files=files,models=models,materials=materials,versions={n:common.importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},expected_new_gib=.4,grid=[.91796875],prime_samples_actual=81600,policy='Existing Jenny396 and earnings12 complete streams, development only; fixed .91796875/smoothing5, no threshold search/new synthesis/training. Continuous81600 zero PCM prime retaining all state and prime events. No numerical equality gate. No final acceptance accessed; single voice cannot establish multi-voice generalization. Prior failed numerical-lock preflights do not bar practical replay. Frozen v30 unchanged.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(common.sha(LOCK))
def run():
    tree=ast.parse(Path(common.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==372:return ast.copy_location(ast.Constant(408),n)
            if n.value==744:return ast.copy_location(ast.Constant(816),n)
            if n.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-nearword-jenny-original-v1'),n)
            return n
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
