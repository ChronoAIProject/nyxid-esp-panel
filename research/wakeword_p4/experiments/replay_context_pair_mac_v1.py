"""Both final paired models on existing consumed Mac12 synthetic fixtures."""
import ast
import importlib.metadata
import json
from pathlib import Path
import sys
import replay_c161_chatterbox_mac_witness_v1 as parent
import export_chatterbox_context_pair_v1 as exporter
R=parent.R
def lock(c):return R/f'4090-context-{c}-mac12-plan-v1.json'
def output(c):return Path(f'.local/wakeword/context-{c}-mac12-v1')
def plan(c):
    original=R/'4090-c161-chatterbox-mac12-plan-v1.json';p=json.loads(original.read_text());e=exporter.report(c);er=json.loads(e.read_text());m=exporter.output(c)/'int8.tflite'
    assert er['all_static_and_runtime_abi_passed'] and parent.sha(m)==er['model']['sha256']
    paths=[original,e,m,Path(__file__),Path(parent.__file__)]
    p['files'].update({str(f):parent.sha(f) for f in paths});p.update(model=str(m),model_sha256=parent.sha(m),condition=c,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},policy='All12 consumed Mac synthetic fixtures, fixed original C161 profile/81600continuous prime/native480. Daniel/Samantha Nyx/Willow/HiESP bare+command. Not P4 execution or new independent exposure; no tuning/training/acceptance. Both paired arms retained, C161 baseline unchanged.')
    with lock(c).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run(c):
    p=json.loads(lock(c).read_text())
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='c161-chatterbox-mac12-result-v1':return ast.copy_location(ast.Constant(f'context-{c}-mac12-result-v1'),n)
            return n
    env=dict(vars(parent),PLAN=lock(c),OUT=output(c),MODEL=Path(p['model']))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':
    action,c=sys.argv[1:];assert c in ['silence','speech'];{'plan':plan,'run':run}[action](c)
