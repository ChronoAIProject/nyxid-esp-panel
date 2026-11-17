"""Reuse original complete consumed replay clocks with Binary40 cache adapter."""
import ast,importlib,json,subprocess,sys
from pathlib import Path
from binary40_stream_predict_v1 import predict
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results')
ARM=sys.argv[-1];assert ARM in ['positive_only','negative_only']
INITIAL={
    'builtin':('replay_daily_language_pair_v1','4090-c161-chatterbox-builtin-replay-lock-v1.json'),
    'daily':('replay_daily_language_pair_v1','4090-daily-language-pair-plan-v1.json'),
    'continuous':('replay_continuous_chatterbox_v1','4090-continuous-chatterbox-plan-v1.json'),
}
COUNTS=dict(primed=372,jenny=408,joe=360,melo=474,mac=12)
def lock(c):return R/f'4090-binary40-carrier-components-{ARM}-{c}-replay-plan-v1.json'
def output(c):return Path(f'.local/wakeword/binary40-carrier-components-{ARM}-{c}-replay-v1')
def source(c):
    if c in INITIAL:return INITIAL[c]
    if c=='melo':return 'collect_melo_development_v2','4090-melo-development-int8-execution-v2.json'
    if c=='mac':return 'replay_c161_chatterbox_mac_witness_v1','4090-c161-chatterbox-mac12-plan-v1.json'
    return 'collect_multispeaker_priming_v1',f'4090-c161-chatterbox-{c}-plan-v1.json'
def plan(c):
    export=R/f'4090-binary40-carrier-components-{ARM}-export-original-v1.json'
    e=json.loads(export.read_text());assert e['all_neural_tensors_integer'] and e['fresh_reset_exact']
    model=Path(e['model']['path']);assert sha(model)==e['model']['sha256']
    name,original=source(c);original=R/original;module=importlib.import_module(name)
    p=json.loads(original.read_text())
    paths=[Path(__file__),Path(__file__).with_name('binary40_stream_predict_v1.py'),Path(module.__file__),original,export,model,R/f'4090-binary40-carrier-components-{ARM}-export-plan-v1.json']
    p['files'].update({str(f):sha(f) for f in paths})
    m=dict(path=str(model),sha256=e['model']['sha256'])
    if c in INITIAL:p['models']=[dict(name='binary40',**m)]
    elif c=='mac':p['model']=str(model)
    elif c=='melo':p['models']=[dict(id='binary40',role='experimental-consumed-development',**m)]
    else:p['models']=[dict(target='hey_nyx',condition='binary40',**m)]
    p.update(span_arm=ARM,schema=f'binary40-{c}-replay-plan-v1',expected_new_gib=.2,grid=[.91796875],
             policy='One final paired-carrier arm model on complete existing consumed sources; arm identity separately recorded, legacy binary40 keys retained only for collector compatibility; original frontend/PCM/81600continuousprime/480sampleclock/finish handling and five-code event profile unchanged. Explicit caches reset only at full-stream start; signedprobability+128 exact lattice translation. Preserve all raw scores/events, compare authenticated saved C161 without reinference. Reused backgrounds never additive, constructednearwords diagnostic, no devtraining/thresholdselection/acceptance or P4claim.')
    with lock(c).open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run(c):
    name,_=source(c);module=importlib.import_module(name)
    # Compile only the existing run function with explicit globals. This keeps
    # original PCM, frame preparation, priming, timing and trace fields intact.
    node=next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class ReceiptNames(ast.NodeTransformer):
        def visit_Constant(self,n):
            changes={'multispeaker-existing-priming-original-v1':f'binary40-{c}-original-v1','melo-development-int8-original-v2':'binary40-melo-original-v1','c161-chatterbox-mac12-result-v1':'binary40-mac-original-v1'}
            if isinstance(n.value,str) and n.value in changes:return ast.copy_location(ast.Constant(changes[n.value]),n)
            if c in ['primed','jenny','joe'] and n.value==372:return ast.copy_location(ast.Constant(COUNTS[c]),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict':
                for k in n.keywords:
                    if k.arg=='models' and isinstance(k.value,ast.Constant):k.value=ast.Constant(1)
                    if k.arg=='traces' and isinstance(k.value,ast.Constant) and c in COUNTS:k.value=ast.Constant(COUNTS[c])
            return n
    env=dict(vars(module),PLAN=lock(c),LOCK=lock(c),OUT=output(c),predict=predict,
             MODEL=Path(json.loads(lock(c).read_text()).get('model','unused')))
    code=ast.fix_missing_locations(ast.Module(body=[ReceiptNames().visit(node)],type_ignores=[]))
    exec(compile(code,str(__file__),'exec'),env)
    env['run']()
    result=output(c)/'result.json';d=json.loads(result.read_text())
    receipt=dict(schema='binary40-replay-execution-v1',cohort=c,plan_sha256=sha(lock(c)),result_sha256=sha(result),
                 adapter_sha256=sha(Path(__file__).with_name('binary40_stream_predict_v1.py')),
                 scope='Original full replay function with explicit-cache predict adapter and receipt/count substitutions only. Host evidence, no P4.',completed=True)
    (output(c)/'binary40-execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
    action,c=sys.argv[1:3];assert c in {*INITIAL,*COUNTS}
    {'plan':plan,'run':run}[action](c)
