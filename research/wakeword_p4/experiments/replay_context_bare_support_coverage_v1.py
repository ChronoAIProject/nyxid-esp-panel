"""Full consumed coverage for bare-support final model; no baseline reinference."""
import ast,importlib,json
from pathlib import Path
import sys
import export_chatterbox_context_pair_v1 as exporter
import export_chatterbox_bare_support_v1 as daily_exporter
R=exporter.R
COUNTS=dict(primed=372,jenny=408,joe=360,melo=474,mac=12)
def lock(c):return R/f'4090-context-bare-support-{c}-coverage-plan-v1.json'
def output(c):return Path(f'.local/wakeword/context-bare-support-{c}-coverage-v1')
def module(c):
    return importlib.import_module('collect_melo_development_v2' if c=='melo' else 'replay_c161_chatterbox_mac_witness_v1' if c=='mac' else 'collect_multispeaker_priming_v1')
def plan(c):
    export=exporter.report('bare-support')
    if not export.exists():print('not_ready: final bare-support export absent');return 2
    er=json.loads(export.read_text());assert er['all_static_and_runtime_abi_passed']
    model=Path(er['model']['path']);assert exporter.exercised.sha(model)==er['model']['sha256']
    original=R/('4090-melo-development-int8-execution-v2.json' if c=='melo' else '4090-c161-chatterbox-mac12-plan-v1.json' if c=='mac' else f'4090-c161-chatterbox-{c}-plan-v1.json')
    p=json.loads(original.read_text());paths=[Path(daily_exporter.__file__),Path(exporter.__file__),Path(__file__),Path(module(c).__file__),export,exporter.lock('bare-support'),model,original]
    p['files'].update({str(f):exporter.exercised.sha(f) for f in paths})
    if c=='mac':p['model']=str(model)
    elif c=='melo':p['models']=[dict(id='context-bare-support',path=str(model),sha256=er['model']['sha256'],role='experimental-consumed-development')]
    else:p['models']=[dict(target='hey_nyx',condition='context-bare-support',path=str(model),sha256=er['model']['sha256'])]
    p.update(schema=f'context-bare-support-{c}-coverage-plan-v1',expected_new_gib=.2,grid=[.91796875],
        policy='Single fixedfinal bare-support arm, all existing consumed inputs, original PCM/continuous81600prime/native480 profile unchanged. No tuning, new generation, training, acceptance or baseline reinference. Preserve raw UINT8, prime and finish clocks. Reused backgrounds nonadditive, nearwords diagnostic, Joe missing carrier remains whole-utterance-unclassified. Report every outcome; no automatic promotion or P4/human/retained-command qualification.')
    with lock(c).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run(c):
    parent=module(c)
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            replacements={'multispeaker-existing-priming-original-v1':f'context-bare-support-{c}-coverage-original-v1','melo-development-int8-original-v2':'context-bare-support-melo-coverage-original-v1','c161-chatterbox-mac12-result-v1':'context-bare-support-mac12-result-v1'}
            if isinstance(n.value,str) and n.value in replacements:return ast.copy_location(ast.Constant(replacements[n.value]),n)
            if c in ['primed','jenny','joe'] and n.value==372:return ast.copy_location(ast.Constant(COUNTS[c]),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict':
                for k in n.keywords:
                    if k.arg=='models' and isinstance(k.value,ast.Constant):k.value=ast.Constant(1)
                    if k.arg=='traces' and isinstance(k.value,ast.Constant):k.value=ast.Constant(COUNTS[c])
            return n
    env=dict(vars(parent),LOCK=lock(c),PLAN=lock(c),OUT=output(c),MODEL=Path(json.loads(lock(c).read_text()).get('model','unused')))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    action,c=sys.argv[1:];assert c in COUNTS
    raise SystemExit({'plan':plan,'run':run}[action](c))
