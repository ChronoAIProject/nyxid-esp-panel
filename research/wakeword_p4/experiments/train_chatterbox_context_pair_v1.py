"""Two fixed-budget arms; reuse original exercised GPU update implementation."""
import ast
import json
from pathlib import Path
import sys
import train_causal_source_balanced_v1 as parent
from causal_chatterbox_context_batch_v1 import Sampler
R=parent.R

def lock(condition):return R/f'4090-context-{condition}-training-plan-v1.json'
def output(condition):return Path(f'.local/wakeword/context-{condition}-training-v1')
def report(condition):return R/f'4090-context-{condition}-training-original-v1.json'

def plan():
    original=R/'4090-c161-chatterbox-training-plan-v1.json'
    base=json.loads(original.read_text());cache=Path('.local/wakeword/chatterbox-context-cache-v1');result=json.loads((cache/'result.json').read_text())
    assert result['completed'] and result['pairs']==168 and result['paired_geometry_exact']
    audit=R/'4090-chatterbox-context-batch-audit-v1.json';assert json.loads(audit.read_text())['all_passed']
    for name,info in result['files'].items():assert parent.sha(cache/name)==info['sha256']
    extra=[Path(__file__),Path(__file__).with_name('causal_chatterbox_context_batch_v1.py'),original,audit,cache/'result.json',R/'4090-chatterbox-context-cache-plan-v1.json',Path('research/wakeword_p4/training/gpu_python.py')]+[cache/n for n in result['files']]
    for condition in ['silence','speech']:
        p=json.loads(json.dumps(base));p['files'].update({str(f):parent.sha(f) for f in extra})
        p.update(schema=f'context-{condition}-training-plan-v1',condition=condition,expected_new_gib=1.0,hypothesis='Paired same-source/same-phase/mask history intervention. Speech versus silence preceding waveform changes frontend-conditioned features and retained model input context jointly. Not an isolated model or frontend causal attribution.',policy='Two sequential arms silence then speech, original C161 initialization/seed726/freshAdam1e-5/frozenBN/20000updates. Both preserve bc01 target draws and120other batch slots; gap RNG730 paired. Four contextual positive/four contextual negative slots. Training-only84targets x2gaps with declared >=360frame history crop after continuous frontend extraction, whole target/tail retained. Same masks and phases between arms; no consumed builtin development audio or acceptance. Fixed final checkpoint only, no threshold/LR/seed/checkpoint rescue. Original export representative retained unchanged as calibration control (not this arms first actual batch); zero/26 sentinels, no calibration tuning. Export each final once, report both outcomes without selection. C161 baseline retained, no human/P4 qualification presumed.')
        with lock(condition).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')

def run(condition):
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant(report(condition).name),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant(f'context-{condition}-training-original-v1'),n)
            return n
    env=dict(vars(parent));env.update(LOCK=lock(condition),OUT=output(condition),Sampler=lambda seed:Sampler(seed,condition))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:assert sys.argv[1]=='run' and sys.argv[2] in ['silence','speech'];run(sys.argv[2])
