"""Both final paired models on consumed Legacy/Jenny/Joe; no tuning."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import collect_multispeaker_priming_v1 as common
import export_chatterbox_context_pair_v1 as exporter
from uint8_event_policy_v1 import event_steps
R=common.R
COUNTS={'primed':372,'jenny':408,'joe':360}
def lock(cohort):return R/f'4090-context-pair-{cohort}-coverage-plan-v1.json'
def output(cohort):return Path(f'.local/wakeword/context-pair-{cohort}-coverage-v1')
def plan(cohort):
    original=R/f'4090-c161-chatterbox-{cohort}-plan-v1.json';p=json.loads(original.read_text());models=[];paths=[original,Path(__file__),Path(common.__file__)]
    for condition in ['silence','speech']:
        epath=exporter.report(condition);e=json.loads(epath.read_text());assert e['all_static_and_runtime_abi_passed'];m=e['model'];assert common.sha(m['path'])==m['sha256']
        models.append(dict(target='hey_nyx',condition=f'context-{condition}',path=m['path'],sha256=m['sha256']));paths.extend([epath,Path(m['path'])])
    p['files'].update({str(f):common.sha(f) for f in paths})
    p.update(schema=f'context-pair-{cohort}-coverage-plan-v1',models=models,grid=[.91796875],expected_new_gib=.2,policy='Both final paired models on all existing consumed inputs. Original whole PCM/continuous81600prime/native480/event profile unchanged; no fitting, newTTS, threshold grid or protected acceptance. Saved C161 reports/traces authenticated separately without baseline inference. Constructed nearwords diagnostic, not zero-required rejection. Background reused/nonadditive. Both arms retained, C161 baseline. Generated boundary references not phonetic truth.')
    with lock(cohort).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run(cohort):
    fn=next(n for n in ast.parse(Path(common.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==372:return ast.copy_location(ast.Constant(COUNTS[cohort]),n)
            if n.value=='multispeaker-existing-priming-original-v1':return ast.copy_location(ast.Constant(f'context-pair-{cohort}-coverage-original-v1'),n)
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='completed' for k in n.keywords):
                for k in n.keywords:
                    if k.arg=='traces':k.value=ast.Constant(2*COUNTS[cohort])
            return n
    env=dict(vars(common),LOCK=lock(cohort),OUT=output(cohort));exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
def score(cohort):
    out=output(cohort);p=json.loads(lock(cohort).read_text());result=json.loads((out/'result.json').read_text());assert result['all_passed'] and result['execution_sha256']==common.sha(lock(cohort))
    base=Path(f'.local/wakeword/causal-source-balanced-{cohort}-v1');br=json.loads((base/'result.json').read_text());assert br['all_passed']
    reports={}
    for name,root,receipt in [('c161',base,br)]+[(f'context-{c}',out,result) for c in ['silence','speech']]:
        modelname='hey_nyx-source-balanced' if name=='c161' else f'hey_nyx-{name}'
        m,=[m for m in receipt['models'] if m['model']==modelname];f=Path(m['path']);assert common.sha(f)==m['sha256']
        report=json.loads(f.read_text());assert report['all_passed'] and len(report['records'])==COUNTS[cohort];reports[name]=report['records']
    groups={};records=[];background=dict(reused_hours=0,events={n:0 for n in reports})
    for i in range(COUNTS[cohort]):
        original=reports['c161'][i];row=original['record'];phase=original['phase'];events={}
        for name,rows in reports.items():
            rec=rows[i];assert rec['record']==row and rec['phase']==phase and rec['index']==original['index'] and rec['prime_samples']==81600
            f=Path(rec['trace_path']);assert common.sha(f)==rec['trace_sha256']
            with np.load(f) as z:
                assert z['uint8'].dtype==np.uint8;clock=z['real_block_return_sample']
                events[name]=[dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(z['uint8'],.91796875)]
        if phase=='background':
            background['reused_hours']+=row['samples']/16000/3600
            for name in reports:background['events'][name]+=sum(e['relative_sample']>0 for e in events[name])
        else:
            target=row['target_id']=='hey_nyx';key=('target/' if target else 'negative/'+row['target_id']+'/')+row['carrier']
            g=groups.setdefault(key,dict(clips=0,hits={n:0 for n in reports},gained={n:0 for n in reports if n!='c161'},lost={n:0 for n in reports if n!='c161'}));g['clips']+=1
            hits={n:any(e['relative_sample']>0 and (not target or e['relative_sample']>=row['lead_samples']) for e in es) for n,es in events.items()}
            for n in reports:g['hits'][n]+=hits[n]
            for n in g['gained']:g['gained'][n]+=hits[n] and not hits['c161'];g['lost'][n]+=hits['c161'] and not hits[n]
        records.append(dict(phase=phase,index=original['index'],source=row,events=events))
    dest=R/f'4090-context-pair-{cohort}-coverage-comparison-v1.json'
    with dest.open('x') as f:f.write(json.dumps(dict(cohort=cohort,threshold=.91796875,groups=groups,background=background,records=records,threshold_selected=False,acceptance_accessed=False,limits='Consumed inputs; synthetic generated boundaries not phonetic truth. Target hit requires event at/after source onset; other negatives require real event. Full prime events retained. Reused background duration nonadditive. No promotion.'),indent=2)+'\n')
    print(json.dumps(dict(groups=groups,background=background),indent=2))
if __name__=='__main__':
    action,cohort=sys.argv[1:];assert cohort in COUNTS;{'plan':plan,'run':run,'score':score}[action](cohort)
