"""Single equal-budget lexical-negative intervention on original initialization."""
import ast,json
from pathlib import Path
import sys
import train_causal_source_balanced_v1 as parent
from causal_chatterbox_daily_context_batch_v1 import Sampler,CACHE
R=parent.R;LOCK=R/'4090-daily-context-training-plan-v1.json';OUT=Path('.local/wakeword/daily-context-training-v1');REPORT='4090-daily-context-training-original-v1.json'
def plan():
    original=R/'4090-context-mixed-training-plan-v1.json';p=json.loads(original.read_text())
    audit=R/'4090-chatterbox-daily-context-batch-audit-v1.json';assert json.loads(audit.read_text())['all_passed']
    cache=json.loads((CACHE/'result.json').read_text());assert cache['completed'] and cache['pairs']==342
    paths=[Path(__file__),Path(__file__).with_name('causal_chatterbox_daily_context_batch_v1.py'),Path(__file__).with_name('check_chatterbox_daily_context_batch_v1.py'),original,audit,CACHE/'result.json',R/'4090-chatterbox-daily-context-cache-plan-v1.json',R/'4090-chatterbox-daily-training-admission-v1.json']
    for name,m in cache['files'].items():
        f=CACHE/name;assert parent.sha(f)==m['sha256'];paths.append(f)
    p['files'].update({str(f):parent.sha(f) for f in paths})
    p.update(schema='daily-context-training-plan-v1',condition='daily-context',
        hypothesis='Replacing narrow four-template synthetic negative support by admitted everyday lexical contexts, with same positive/parent presentations and negative count, may reduce daily conflicts without sacrificing wake coverage. Target durations/masks/phase/prefix distribution change; not lexical-only causal isolation.',
        policy='One RTX4090 original shared C161 initialization run, seed726/freshAdam1e-5/frozenBN/20000updates/finalonly. No candidate finetuning. All124other inputs/masks/source draws exact in12batch checks; replace four negative slots retaining each original voice/gap and mixed2silence+2speech budget. SeparateRNG732: uniform6categories then uniform admitted samevoice/gap target.171trainingonly ASR-consistent negatives;21ambiguous quarantined unchanged. New data designed after consumed six-cups observation, not independent confirmation or new human voices. Same scalar loss/graph/calibration representative control. No development voice/audio admission, acceptance, threshold/LR/seed/checkpoint/calibration rescue. One final INT8 export, all consumed cohorts plus training-fit evidence retained. Report hits lost as well as conflicts removed; no promotion based on fewer nearwords. C161 baseline remains, P4/firstcommand/resources separate incomplete evidence.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant(REPORT),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant('daily-context-training-original-v1'),n)
            return n
    env=dict(vars(parent),LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
