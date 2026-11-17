"""Lock all48 training-only clips for pinned CUDA small.en transcript review."""
import ast,json
from pathlib import Path
import sys
import audit_chatterbox_training_asr_v1 as parent
DATA=Path('.local/wakeword/chatterbox-bare-training-v1')
PLAN=Path('research/wakeword_p4/results/4090-chatterbox-bare-training-asr-plan-v1.json')
OUT=Path('.local/wakeword/chatterbox-bare-training-asr-v1')
RUNNER=Path(__file__).with_name('audit_daily_training_asr_runner_v1.py')
def lock():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='lock')
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==96:return ast.copy_location(ast.Constant(48),n)
            return n
    env=dict(vars(parent),DATA=DATA,PLAN=PLAN,RUNNER=RUNNER)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Count().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['lock']()
    p=json.loads(PLAN.read_text())
    for f in [Path(__file__),Path(parent.__file__),parent.RUNNER]:p['files'][str(f)]=parent.sha(f)
    p['policy']='All48 provisional positive clips once; training-only four conditioning identities. small.en CUDAfloat16/beam3/temp0/no prompt/noVAD/independent segments, CPU2threads. No wake scoring, relabeling or acceptance. Normalize alphabetic words exactly as historical auditor; no automatic positive admission; transcript mismatch/target pronunciation ambiguity quarantined in separate later admission, no retry. Bare-positive support intervention after consumed exposure audit; not independent validation. Text agreement not phonetic truth.'
    PLAN.write_text(json.dumps(p,indent=2)+'\n')
def run():
    parent.DATA=DATA;parent.PLAN=PLAN;parent.OUT=OUT;parent.RUNNER=RUNNER;parent.run()
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
