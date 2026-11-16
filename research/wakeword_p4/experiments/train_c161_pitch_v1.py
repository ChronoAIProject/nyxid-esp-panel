"""Single fixed-budget C161-control recipe with training waveform pitch intervention."""
import ast,json,sys
from pathlib import Path
import train_causal_source_balanced_v1 as parent
from causal_pitch_cache_batch_v1 import Sampler
R=parent.R;LOCK=R/'4090-c161-pitch-training-plan-v1.json';OUT=Path('.local/wakeword/c161-pitch-training-v1')
def plan():
    original=R/'4090-causal-source-balanced-plan-v1.json';p=json.loads(original.read_text())
    cache=Path('.local/wakeword/c161-pitch-cache-v2');result=cache/'result.json';cr=json.loads(result.read_text());assert cr['records']==31644
    for name,h in cr['files'].items():assert parent.sha(cache/name)==h
    batch=R/'4090-c161-pitch-batch-check-v1.json';b=json.loads(batch.read_text());assert b['source_schedule_unchanged'] and b['supervision_unchanged']
    files=[Path(__file__).resolve(),Path(parent.__file__),original,result,cache/'records.jsonl',cache/'features.bin',batch,Path(__file__).with_name('causal_pitch_cache_batch_v1.py'),Path(__file__).with_name('check_c161_pitch_batch_v1.py')]
    p['files'].update({str(f):parent.sha(f) for f in files})
    p.update(schema='c161-pitch-training-plan-v1',hypothesis='Fixed50% waveform pitch derivatives in original8broad positive+8sourcebalanced negative slots may improve voice coverage; no efficacy assumed. Same originalinit,seed726,Adam1e-5,BNfreeze,20000updates/finalcheckpoint as C161, all source schedule/labels/loss unchanged. Original C161 is preserved, no development example training.',policy='Pitch RNG728 independent of parent source RNG. Fixed +/-200cents keyed source/variant, .8headroom and<=1sample end rounding recorded in cache. No threshold/LR/seed/checkpoint search or final access. Evaluate fixed.91796875/smoothing5 using saved C161 baseline on consumed legacy/Jenny/Joe/Melo/Mac12 protocols; no candidate promotion from rare nearword total alone.',expected_new_gib=1.0)
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    tree=ast.parse(Path(parent.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant('4090-c161-pitch-training-original-v1.json'),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant('c161-pitch-training-original-v1'),n)
            return n
    env=dict(vars(parent));env.update(LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
