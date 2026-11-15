"""Same deployed event function on all378complete training bare waveforms.

Do not equate calibration-window classification with actual wake events. Reuse
the execution-bound complete-wave collector unchanged via narrow AST adaptation.
"""
import ast
import json
from pathlib import Path
import sys
import collect_matchbox_consumed_melo_v1 as common
R=common.R;BASE=common.BASE;sha=common.sha
LOCK=R/'4090-matchbox-training-stream-plan-v1.json';OUT=BASE/'matchbox-training-stream-v1'
SOURCE=Path(common.__file__)
def plan():
    parent=common.LOCK;p=json.loads(parent.read_text());training=R/'4090-matchbox-window-training-plan-v1.json';t=json.loads(training.read_text());original=BASE/'speecht5-training-rendered-v1/records.jsonl'
    assert t['steps']==5000;records=[]
    for row in map(json.loads,original.read_text().splitlines()):
        if row['carrier']!='bare':continue
        assert row['role']=='training' and row['passed'] and sha(row['path'])==row['sha256']
        end=row['generated_container_end_sample'];rirend=end+row['rir_tail_samples']
        records.append(dict(row,phase='synthetic',generated_prefix_end_sample=end,full_rir_prefix_end_sample=rirend,full_rir_container_end_sample=rirend,bare_full_container_reference_not_phonetic=True))
    assert len(records)==378 and all(sum(r['phrase_id']==phrase for r in records)==63 for phrase in ['nyx','official','nick','next','mix','muse'])
    p.update(schema='matchbox-training-stream-plan-v1',records=records,policy='Training-only diagnostic, all378complete original SpeechT5 bare presentations under exact same frozenmodel/raw20480context/MFCC/480cadence/81600prime/0.91796875/fiveUINT8/67refractory event function as consumedMelo. No window-only recognition substituted for streamingwake recall. Whole bare container/RIR endpoints are descriptive generation references, not phonetic truth or invented command ends. Reuse execution-bound collector by explicit AST schema/count/report/FAR adaptation; no model/frontend/decoder change. Zero background exposure here, FAR is null; no heldout/freshdev/acceptance/threshold/calibration/retraining/promotion. v30/Mac unchanged.')
    p['files'].update({str(f):sha(f) for f in [Path(__file__),SOURCE,parent,training,original]})
    p['files'].update({r['path']:r['sha256'] for r in records})
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(records=378,plan_sha256=sha(LOCK))),flush=True)
def run():
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run');counts=dict(total=0,common=0,cross=0,report=0,schema=0,far=0,complete=0)
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,node):
            changes={474:(378,'total'),288:(252,'common'),72:(63,'cross'),'4090-matchbox-consumed-melo-original-v1.json':('4090-matchbox-training-stream-original-v1.json','report'),'matchbox-consumed-melo-original-v1':('matchbox-training-stream-original-v1','schema')}
            if type(node.value) in (int,str) and node.value in changes:
                value,key=changes[node.value];counts[key]+=1;return ast.copy_location(ast.Constant(value),node)
            return node
        def visit_keyword(self,node):
            if node.arg=='background_far_upper95':counts['far']+=1;node.value=ast.Constant(None);return node
            if node.arg=='all474streams_complete':counts['complete']+=1;node.arg='all378streams_complete'
            if node.arg=='consumed_diagnostic_only':node.arg='training_diagnostic_only'
            return self.generic_visit(node)
    function=Adapter().visit(function);assert counts==dict(total=1,common=1,cross=1,report=1,schema=1,far=1,complete=1),counts
    strata={speaker+'/bare':9 for speaker in ['awb','bdl','clb','jmk','ksp','rms','slt']}
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT,STRATA=strata)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE),'exec'),env);env['run']()
if __name__=='__main__':
    action=sys.argv[1];{'plan':plan,'run':run}[action]()
