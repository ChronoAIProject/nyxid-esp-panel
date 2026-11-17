"""Single equal-budget lexical-negative intervention on original initialization."""
import ast,json
from pathlib import Path
import sys
import train_causal_source_balanced_v1 as parent
from causal_chatterbox_bare_support_batch_v1 import Sampler,CACHE
R=parent.R;LOCK=R/'4090-bare-support-training-plan-v1.json';OUT=Path('.local/wakeword/bare-support-training-v1');REPORT='4090-bare-support-training-original-v1.json'
def plan():
    original=R/'4090-daily-context-training-plan-v1.json';p=json.loads(original.read_text())
    audit=R/'4090-chatterbox-bare-support-batch-audit-v1.json';a=json.loads(audit.read_text());assert len(a['paired_batches'])==12 and all(x['unchanged_slots']==126 for x in a['paired_batches'])
    cache=json.loads((CACHE/'result.json').read_text());assert cache['completed'] and cache['pairs']==46
    paths=[Path(__file__),Path(__file__).with_name('causal_chatterbox_bare_support_batch_v1.py'),Path(__file__).with_name('check_chatterbox_bare_support_batch_v1.py'),original,audit,CACHE/'result.json',R/'4090-chatterbox-bare-context-cache-plan-v1.json',R/'4090-chatterbox-bare-training-admission-v1.json']
    for name,m in cache['files'].items():
        f=CACHE/name;assert parent.sha(f)==m['sha256'];paths.append(f)
    p['files'].update({str(f):parent.sha(f) for f in paths})
    p.update(schema='bare-support-training-plan-v1',condition='bare-support',
        hypothesis='New provisional bare training support with exact voice/history quotas may improve normal wake coverage without losing command hits. Two original broad bare slots are replaced; reduced broad diversity is an explicit tradeoff, not a pure phonetic causal isolation.',
        policy='One RTX4090 original shared C161 initialization, seed726/freshAdam1e-5/frozenBN/20000updates/finalonly, not candidate finetune.126inputs/masks/descriptors exact in12paired batches versus daily-context; preserve all4original contextual positive and4dailynegative slots. Replace2remaining broad bare slots with23provisional ASR-consistent/mechanically aligned bare sources,25quarantined excluded. Same4trainingreferences, separateRNG733,2newvoices eachbatch/one speech+one silence, equal8voice/history combinations each4batches. Hypothesis motivated after consumed audits, not independent validation. Text/alignment is not phonetic truth. No development audio/voice/acceptance admission, no threshold/LR/seed/checkpoint/calibration rescue. Same scalar loss/graph/original representative control, single finalINT8export. Complete consumed coverage and old/new fit, C161/mixed/daily-context comparisons with hits lost and everyday conflicts jointly. No promotion from constructed nearword counts. C161 remains baseline, P4 numeric/firstcommand/resources/stability separately incomplete.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant(REPORT),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant('bare-support-training-original-v1'),n)
            return n
    env=dict(vars(parent),LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
