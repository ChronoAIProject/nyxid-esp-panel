"""Both final paired models on all consumed Melo474; saved C161 comparison."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import collect_melo_development_v2 as parent
import export_chatterbox_context_pair_v1 as exporter
from uint8_event_policy_v1 import event_steps
R=parent.R;LOCK=R/'4090-context-pair-melo-plan-v1.json';OUT=Path('.local/wakeword/context-pair-melo-v1');BASE=Path('.local/wakeword/melo-development-int8-v2/c161')

def plan():
    source=R/'4090-melo-development-int8-execution-v2.json';p=json.loads(source.read_text());models=[];paths=[Path(__file__),Path(parent.__file__),source]
    for condition in ['silence','speech']:
        report=exporter.report(condition)
        if not report.exists():print(json.dumps(dict(status='not_ready',missing=str(report),model_calls=0)));return 2
        e=json.loads(report.read_text());assert e['all_static_and_runtime_abi_passed']
        model=exporter.output(condition)/'int8.tflite';assert parent.sha(model)==e['model']['sha256']
        models.append(dict(id=f'context-{condition}',path=str(model),sha256=e['model']['sha256'],role='paired-experimental-development'))
        paths.extend([report,exporter.lock(condition),model])
    baseline=BASE.parent/'result.json';b=json.loads(baseline.read_text());assert b['all_passed'] and parent.sha(BASE/'records.jsonl')==b['record_files']['c161']
    paths.extend([baseline,BASE/'records.jsonl'])
    p['files'].update({str(f):parent.sha(f) for f in paths})
    p.update(schema='context-pair-consumed-melo-plan-v1',models=models,expected_new_gib=.15,reference_threshold=.91796875,policy='Both paired fixedfinal models on all474 consumed Melo inputs432synthetic+42complete reused background. Fresh per-recording frontend/interpreter with continuous81600prime, original native480 clocks, UINT8 retained. C161 saved trace comparison authenticated separately. Fixed .91796875/smoothing5/refractory67/half rearm; no threshold grid, acceptance or training. EN-BR is British. All background hours are reused, not additive independent exposure. Original role metadata retained for identity only; no longer unconsumed. Report both arms, baseline C161 unchanged.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict':
                for k in n.keywords:
                    if k.arg=='models' and isinstance(k.value,ast.Constant):k.value=ast.Constant(2)
            return n
        def visit_Constant(self,n):
            if n.value=='melo-development-int8-original-v2':return ast.copy_location(ast.Constant('context-pair-consumed-melo-original-v1'),n)
            return n
    env=dict(vars(parent),LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()

def score():
    result=json.loads((OUT/'result.json').read_text());assert result['all_passed'] and result['counts']=={'context-silence':474,'context-speech':474}
    baseline=json.loads((BASE.parent/'result.json').read_text());assert baseline['all_passed']
    paths={'c161':BASE/'records.jsonl',**{n:OUT/n/'records.jsonl' for n in result['counts']}}
    assert parent.sha(paths['c161'])==baseline['record_files']['c161']
    for n in result['counts']:assert parent.sha(paths[n])==result['record_files'][n]
    ledgers={n:[json.loads(s) for s in f.read_text().splitlines()] for n,f in paths.items()};assert all(len(v)==474 for v in ledgers.values())
    groups={};records=[];background=dict(reused_hours=0,events={n:0 for n in paths})
    for index in range(474):
        originals={n:v[index] for n,v in ledgers.items()};source=originals['c161']['record'];events={}
        for n,rec in originals.items():
            assert rec['index']==index and rec['record']==source and rec['prime_samples']==81600
            path=Path(rec['trace_path']);assert parent.sha(path)==rec['trace_sha256']
            with np.load(path) as z:
                assert z['uint8'].dtype==np.uint8
                clock=z['block_return_sample']-81600
                events[n]=[dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(z['uint8'],.91796875)]
        real={n:[e for e in v if e['relative_sample']>0] for n,v in events.items()}
        if source['phase']=='background':
            background['reused_hours']+=source['samples']/16000/3600
            for n in paths:background['events'][n]+=len(real[n])
        else:
            key=source['voice_id'].split(':')[-1]+'/'+source['phrase_id']+'/'+source['carrier']
            g=groups.setdefault(key,dict(clips=0,hits={n:0 for n in paths},paired_against_c161={n:dict(gained=0,lost=0) for n in result['counts']}));g['clips']+=1
            for n in paths:g['hits'][n]+=bool(real[n])
            for n in result['counts']:
                g['paired_against_c161'][n]['gained']+=bool(real[n]) and not bool(real['c161'])
                g['paired_against_c161'][n]['lost']+=bool(real['c161']) and not bool(real[n])
        records.append(dict(index=index,source=source,events=events))
    report=dict(inputs={str(f):parent.sha(f) for f in paths.values()},threshold=.91796875,smoothing_steps=5,groups=groups,background=background,records=records,threshold_selected=False,acceptance_accessed=False,policy='All consumed inputs, raw-byte event arithmetic. EN-BR British. Background reused/nonadditive. Container events not phonetic truth. No human/P4 qualification or automatic promotion.')
    dest=R/'4090-context-pair-melo-fixed-comparison-v1.json'
    with dest.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(groups=groups,background=background),indent=2))
if __name__=='__main__':
    action=sys.argv[1];assert action in ['plan','run','score'];raise SystemExit({'plan':plan,'run':run,'score':score}[action]())
