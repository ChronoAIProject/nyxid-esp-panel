"""Fixedfinal causal INT8 and original frozen v30 on identical complete inputs."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import collect_multispeaker_formal_v1 as common
from collect_complete_development_v1 import runtime_identity_files
R=common.R
LOCK=R/'4090-causal-sequence-formal-development-plan-v1.json'
OUT=Path('.local/wakeword/causal-sequence-formal-development-v1')
SOURCE=Path(common.__file__)
sha=common.sha

def plan():
    report=R/'4090-causal-sequence-formal-int8-original-v1.json'
    if not report.exists():
        print(json.dumps(dict(status='not_ready',missing=str(report),model_calls=0)));return 2
    r=json.loads(report.read_text());export_plan=R/'4090-causal-sequence-formal-int8-plan-v1.json'
    assert r['all_static_and_runtime_abi_passed'] and r['repeats_exact'] and not r['recognition_scored'] and r['plan_sha256']==sha(export_plan)
    parent=R/'4090-multispeaker-formal-development-execution-v1.json';p=json.loads(parent.read_text());assert len(p['grid'])==138
    files=dict(p['files']);paths=[Path(__file__),SOURCE,report,export_plan,parent,Path(__file__).with_name('score_causal_sequence_formal_v1.py'),Path('research/wakeword_p4/training/features.py')]+runtime_identity_files()
    artifact=r['model'];assert sha(artifact['path'])==artifact['sha256'];paths.append(Path(artifact['path']))
    baseline=Path('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite');baseline_sha='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    assert baseline.stat().st_size==54440 and sha(baseline)==baseline_sha;paths.append(baseline)
    paths.append(Path('research/wakeword_p4/training/evaluate.py'))
    files.update({str(x):sha(x) for x in paths})
    for path,h in files.items():assert sha(path)==h,path
    lock=dict(schema='causal-sequence-formal-development-plan-v1',files=files,models=[dict(target='hey_nyx',condition='causal',path=artifact['path'],sha256=artifact['sha256']),dict(target='hey_nyx',condition='v30-frozen',path=str(baseline),sha256=baseline_sha)],materials=p['materials'],grid=p['grid'],versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},threshold_selected=False,acceptance_scored=False,retries=0,limits='One fixedfinal20000 causal fullsequence/onset INT8 candidate plus original54440B frozenv30 bytes, same372consumed complete inputs and138grid. Cachedv30 legacy traces omitted finalpartial3framecall, so re-run exact same complete-input protocol instead of silently comparing old numbers. Baseline138grid is secondary UINT8 protocol diagnosis only; frozenv30 original threshold.91796875/float64 convolution policy separately retained. Unchanged rawUINT8 smoothing5/67cooldown/halfthresholdrearm. Coldstartup primary, no priming/readiness suppression bundled with candidate; full startup/tail/every first phrase kept. Existingcomplete finalhop/featurezero finish only at end. Twofreshinterpreters per whole input, allrawcodes/clocks/quantization retained. No floatcomparison, fullbatchaudit or numericalagreement prerequisite. Diagnosis only, not thresholdselection/independentacceptance/qualification. Preserve v30 and all failures; no freshdev or acceptance here.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),models=2,inputs=372)))
    return 0

def run():
    function=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    env=dict(vars(common));env.update(LOCK=LOCK,OUT=OUT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE),'exec'),env)
    env['run']()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();raise SystemExit(plan() if a.action=='plan' else run())
