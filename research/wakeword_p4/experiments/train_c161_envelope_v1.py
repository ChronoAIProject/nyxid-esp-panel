"""Original C161 fixed-budget training with one bounded envelope intervention."""
import ast,json,sys
from pathlib import Path
import train_causal_source_balanced_v1 as parent
from causal_envelope_cache_batch_v1 import Sampler,CACHE
R=parent.R;LOCK=R/'4090-c161-envelope-training-plan-v1.json';OUT=Path('.local/wakeword/c161-envelope-training-v1')
def plan():
    original=R/'4090-causal-source-balanced-plan-v1.json';p=json.loads(original.read_text())
    result=CACHE/'result.json';cr=json.loads(result.read_text());assert cr['records']==31644
    for name,h in cr['files'].items():assert parent.sha(CACHE/name)==h
    batch=R/'4090-c161-envelope-batch-check-v1.json';b=json.loads(batch.read_text());assert b['all_row_hashes_verified']==31644 and b['source_schedule_unchanged'] and b['supervision_unchanged'] and b['unselected_tensors_unchanged']
    files=[Path(__file__),Path(parent.__file__),original,result,CACHE/'records.jsonl',CACHE/'features.bin',batch,Path(__file__).with_name('causal_envelope_cache_batch_v1.py'),Path(__file__).with_name('check_c161_envelope_batch_v1.py'),Path(__file__).with_name('supervise_c161_envelope_v1.py'),R/'4090-c161-envelope-cache-plan-v1.json']
    p['files'].update({str(f):parent.sha(f) for f in files})
    p.update(schema='c161-envelope-training-plan-v1',hypothesis='Fullwave spectral-envelope perturbation may reduce dependence on envelope coloration while preserving original class/source support; harmonic estimator pilot differs from whole-spectrum pitch shift, no phonetic or efficacy claim. Original8broad positive and8sourcebalanced negative eligible slots, original source schedule/masks/loss/architecture unchanged.',policy='Single originalsharedinit/seed726/freshAdam1e-5/frozenBN/20000updates/finalonly RTX4090, not candidate finetune. SeparateaugmentationRNG734 Bernoulli50%; fixed.9/1.1 per sourcepair/variant independent of text/label. Originalfullwave/rmsmatch/ratiocap/limiter effects explicit, no new voices or quarantined admission. No coefficient/threshold/LR/seed/checkpoint/calibration rescue, no heldout access or developer voices. Compare saved C161 on complete consumed Legacy/Jenny/Joe/Melo/Mac and current builtin/daily/continuous, joint normalhit losses/command/crossvoice/everydayconflicts/background, constructednearwords diagnostic. Single finalINT8 export same representative control. C161 retained unless full evidence supports replacement; P4 numeric/firstcommand/resources/stability separate.',expected_new_gib=1.0)
    assert p['steps']==20000
    with LOCK.open('x') as f:json.dump(p,f,indent=2)
def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            replacements={'4090-causal-source-balanced-original-v1.json':'4090-c161-envelope-training-original-v1.json','causal-source-balanced-original-v1':'c161-envelope-training-original-v1'}
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            return n
    env=dict(vars(parent),LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
