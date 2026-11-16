"""Single fixed-budget C161-control recipe with conditioned Chatterbox source expansion."""
import ast,json,sys
from pathlib import Path
import train_causal_source_balanced_v1 as parent
from causal_chatterbox_batch_v1 import Sampler
R=parent.R;LOCK=R/'4090-c161-chatterbox-training-plan-v1.json';OUT=Path('.local/wakeword/c161-chatterbox-training-v1')
def plan():
    original=R/'4090-causal-source-balanced-plan-v1.json';p=json.loads(original.read_text())
    cache=Path('.local/wakeword/chatterbox-training-cache-v1');result=cache/'result.json';cr=json.loads(result.read_text());assert cr['records']==84
    for name,h in cr['files'].items():assert parent.sha(cache/name)==h
    batch=R/'4090-chatterbox-training-batch-check-v1.json';b=json.loads(batch.read_text());assert b['unchanged_other120_exact'] and b['masks_valid']
    files=[Path(__file__).resolve(),Path(parent.__file__),original,result,cache/'records.json',cache/'features.bin',batch,Path(__file__).with_name('causal_chatterbox_batch_v1.py'),R/'4090-chatterbox-training-admission-v1.json',R/'4090-chatterbox-training-cache-plan-v1.json']
    p['files'].update({str(f):parent.sha(f) for f in files})
    p.update(schema='c161-chatterbox-training-plan-v1',hypothesis='Four training-only synthetic conditioned Chatterbox voices may expand pronunciation/prosody coverage. Exactly4broadpositive+4sourcebalancednegative slots per128 replaced with independent RNG729. Original32positive total and other120tensors preserved. Source-matched new daily negatives, prefix-timely positives; no efficacy presumed.',policy='Same original C161 initialization, seed726, freshAdam1e-5, frozenBN,20000updates/finalcheckpoint/singleINT8. Not finetuning exported C161.36ASR-consistent aligned positives+48exactnegative wholewaves,12ambiguous positives excluded unchanged. No builtin development voice training or acceptance access. Reused conditioning identities are not independent human speakers. Fixed .91796875/5 replay against C161 consumed diagnostics, including new builtin Chatterbox36 and legacy/Jenny/Joe/Mac; rare nearwords diagnostic not zero requirement. No LR/seed/threshold/checkpoint rescue. Existing training data reused, no fabricated background duration.',expected_new_gib=1.0)
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    tree=ast.parse(Path(parent.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant('4090-c161-chatterbox-training-original-v1.json'),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant('c161-chatterbox-training-original-v1'),n)
            return n
    env=dict(vars(parent));env.update(LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
