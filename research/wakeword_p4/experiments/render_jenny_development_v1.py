"""Separate development render adapter over pinned original complete-source renderer."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from render_hey_broad_training_v1 import versions

RESULTS=Path('research/wakeword_p4/results')
ORIGINAL=Path(__file__).with_name('render_hey_broad_training_v1.py')
PLAN=RESULTS/'4090-jenny-development-render-plan-v1.json'
SCHEDULE=Path('.local/wakeword/jenny-development-render-schedule-v1.jsonl')
OUT=Path('.local/wakeword/jenny-development-rendered-v1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    native=RESULTS/'4090-jenny-development-native-generation-v1.json';audit=RESULTS/'4090-jenny-development-independent-native-audit-v1.json'
    n=json.loads(native.read_text());a=json.loads(audit.read_text())
    assert sha(native)==a['report_sha256'] and n['quality_passes']==a['rows']==132
    records=Path(n['records_path']);assert sha(records)==n['records_sha256']==a['records_sha256']
    rows=[]
    for i,r in enumerate(map(json.loads,records.read_text().splitlines())):
        req=r['request'];assert req['order']==i and r['quality_pass']
        key=dict(engine='author-direct-jenny',conditioning='Jenny-Dioco',carrier=req['carrier'],scale=req['length_scale'])
        group=['bare','light','time','timer'].index(req['carrier'])*3+[.9,1.,1.1].index(req['length_scale'])
        rows.append(dict(order=i,id=f'jenny-development-{i:03}',kind='positive',target=req['phrase'],engine='author-direct-jenny',pair_key=key,
            role='development',text=req['text'],source_path=r['files']['wav']['path'],source_sha256=r['files']['wav']['sha256'],
            source_pcm_sha256=r['pcm_sha256'],source_samples=r['samples'],generated_prefix_end=r['generated_prefix_end_sample'],
            render_seeds=[92826000+group*3+v for v in range(3)]))
    with SCHEDULE.open('x') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    decision=RESULTS/'4090-hey-broad-context-decision-v1.json'
    original_plan=json.loads((RESULTS/'4090-hey-broad-render-plan-v1.json').read_text());assert sha(ORIGINAL)==original_plan['source_sha256']
    d=dict(schema='jenny-development-render-plan-v1',source_sha256=sha(ORIGINAL),
        files={str(p):sha(p) for p in [Path(__file__),ORIGINAL,native,audit,records]},
        context_decision_sha256=sha(decision),versions=versions(),schedule=dict(path=str(SCHEDULE),sha256=sha(SCHEDULE),sources=132,rendered=396),
        lead_samples=48000,tail_samples=16000,conditions=['quiet','25dB_noise','15dB_noise_full_RIR'],
        rir_delays=[0,160,480,960],rir_amplitudes=[1.,.25,-.12,.06],
        policy='Whole original serialized PCM, no speed change/trim. Reuse exact original complete-source render function via separate AST adapter; matched condition seeds across11phrases. Keep full RIR tail and prescribed min(1,.95/peak) gain after noisy mixing only.',
        role='Development only, never training/acceptance. kind positive in renderer means retain generated prefix metadata for every phrase, not a classifier label.',
        scoring_authorized=False,limits='Rendering only. Native source/history limitations retained; independent replay/history/background/scorer admission required before wake inference.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')


def run():
    # Keep original bound renderer unchanged; only output identity and progress
    # constant differ. Original quality checks and complete PCM arithmetic remain.
    tree=ast.parse(ORIGINAL.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Rename(ast.NodeTransformer):
        def visit_Constant(self,node):
            replacements={'hey-broad-render-result-v1':'jenny-development-render-result-v1',
                '4090-hey-broad-render-v1.json':'4090-jenny-development-render-v1.json',106956:396}
            if node.value in replacements:return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    fn=Rename().visit(fn)
    env=dict(__file__=str(ORIGINAL),subprocess=subprocess,json=json,np=np,sf=sf,fftconvolve=fftconvolve,
        Path=Path,PLAN=PLAN,OUT=OUT,RESULTS=RESULTS,sha=sha,versions=versions,hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ORIGINAL),'exec'),env)
    env['run']()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args()
    plan() if a.action=='plan' else run()
