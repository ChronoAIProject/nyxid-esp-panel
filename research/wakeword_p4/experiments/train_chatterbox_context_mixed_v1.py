"""One preregistered mixed-context arm; same original GPU update implementation."""
import ast,json
from pathlib import Path
import sys
import train_causal_source_balanced_v1 as parent
from causal_chatterbox_context_mixed_batch_v1 import Sampler
R=parent.R
LOCK=R/'4090-context-mixed-training-plan-v1.json'
OUT=Path('.local/wakeword/context-mixed-training-v1')
REPORT='4090-context-mixed-training-original-v1.json'
def plan():
    original=R/'4090-context-speech-training-plan-v1.json';p=json.loads(original.read_text())
    audit=R/'4090-context-mixed-batch-audit-v1.json';assert json.loads(audit.read_text())['all_passed']
    paths=[Path(__file__),Path(__file__).with_name('causal_chatterbox_context_mixed_batch_v1.py'),Path(__file__).with_name('check_context_mixed_batch_v1.py'),original,audit]
    p['files'].update({str(f):parent.sha(f) for f in paths})
    p.update(schema='context-mixed-training-plan-v1',condition='mixed',
        hypothesis='At fixed total presentation budget, equal silence/speech context within each label may retain isolated and continuous support. Same target/pair/mask draws and parent inputs as existing arms; not new voices or causal isolation of frontend/model history.',
        policy='One RTX4090 run, original C161 shared initialization (not candidate finetune), seed726/freshAdam1e-5/frozenBN/20000updates/final checkpoint only. Each128batch retains120parent slots; four positive/four negative Chatterbox context slots split exactly2silence+2speech per label by separate RNG731. Parent target and gap draws identical to prior arms. No extra presentations, new TTS, consumed development training, acceptance, threshold/LR/seed/checkpoint rescue. Existing calibration representative retained unchanged as control, final INT8 export once. Report complete consumed fixed-profile suite and training-fit diagnosis, all outcomes including daily conflict/British bare/Joe/Daniel. C161 baseline remains; no automatic promotion, independent voice or P4 claim.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='4090-causal-source-balanced-original-v1.json':return ast.copy_location(ast.Constant(REPORT),n)
            if n.value=='causal-source-balanced-original-v1':return ast.copy_location(ast.Constant('context-mixed-training-original-v1'),n)
            return n
    env=dict(vars(parent),LOCK=LOCK,OUT=OUT,Sampler=Sampler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
