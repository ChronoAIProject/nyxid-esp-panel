"""Existing Joe development only; native full-utterance boundaries, no final access."""
import argparse,ast,json
from pathlib import Path
import collect_multispeaker_priming_v1 as common
R=common.R;LOCK=R/'4090-causal-joe-development-plan-v1.json';OUT=Path('.local/wakeword/causal-joe-development-v1')
def plan():
    root=Path('.local/wakeword/joe-development-render-v1');ledger=root/'records.jsonl';receipt=R/'4090-joe-development-render-summary-v1.json';r=json.loads(receipt.read_text());assert common.sha(ledger)==r['records_sha256'] and r['mechanically_qualified']
    rows=[];files={str(p):common.sha(p) for p in [ledger,receipt,Path(__file__),Path(common.__file__),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path('research/wakeword_p4/training/features.py'),R/'4090-joe-known-source-overlap-supplement-v1.json']}
    for row in map(json.loads,ledger.read_text().splitlines()):
        path=root/row['path'];assert common.sha(path)==row['sha256'];files[str(path)]=row['sha256'];rows.append(dict(row,path=str(path),target_id=row['target'],role='consumed-source development only'))
    assert len(rows)==360
    manifest=Path('.local/wakeword/causal-joe-development-manifest-v1.json')
    with manifest.open('x') as f:f.write(json.dumps(dict(records=rows),indent=2)+'\n')
    files[str(manifest)]=common.sha(manifest);models=[]
    for condition,path,h in [('v30','.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite','04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'),('nearword','.local/wakeword/causal-existing-nearword-int8-v1/int8.tflite','9307e6c13b7bc269f2566f111acdddcd8a2fedac569ce9f5c35b4e9fd030a7d6'),('multivoice','.local/wakeword/causal-multivoice-nearword-int8-v1/int8.tflite','95aba77a68610be66c0032500be3dd31962b8ec02a07e7398c26e38759ada5b3')]:
        assert common.sha(path)==h;files[path]=h;models.append(dict(target='hey_nyx',condition=condition,path=path,sha256=h))
    p=dict(schema='causal-joe-development-plan-v1',files=files,models=models,materials=dict(synthetic=dict(path=str(manifest),sha256=common.sha(manifest),role='existing consumed-source development')),versions={n:common.importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},expected_new_gib=.2,grid=[.91796875],prime_samples_actual=81600,policy='Existing360Joe wholewaves only, three originalmodels, fixed.91796875/smoothing5/continuous81600zeroPCMprime. Historical cold v30 baseline is not reused as primed evidence. No new synthesis/render/training/thresholdsearch or final access. Joe is consumed development, shared parent with protectedMike; not new independent acceptance or threshold-selection data. Whole generated output boundary only; no invented wake-prefix boundary or phonetic truth. Basic original file/interface use, no numerical equality gate. V30 retained.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(common.sha(LOCK))
def run():
    fn=next(n for n in ast.parse(Path(common.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==372:return ast.copy_location(ast.Constant(360),n)
            if n.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant('causal-joe-development-original-v1'),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='completed' for k in n.keywords):
                for k in n.keywords:
                    if k.arg=='models':k.value=ast.Constant(3)
                    if k.arg=='traces':k.value=ast.Constant(1080)
            return n
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT);exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
