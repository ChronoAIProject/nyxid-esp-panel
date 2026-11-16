"""Both paired final INT8 models on consumed fixed-profile diagnostics."""
import importlib
import json
from pathlib import Path
import sys
import export_chatterbox_context_pair_v1 as exporter
R=exporter.R
SOURCES={
 'builtin':('replay_daily_language_pair_v1',R/'4090-c161-chatterbox-builtin-replay-lock-v1.json'),
 'daily':('replay_daily_language_pair_v1',R/'4090-daily-language-pair-plan-v1.json'),
 'continuous':('replay_continuous_chatterbox_v1',R/'4090-continuous-chatterbox-plan-v1.json'),
}
def lock(cohort):return R/f'4090-context-pair-{cohort}-replay-plan-v1.json'
def output(cohort):return Path(f'.local/wakeword/context-pair-{cohort}-replay-v1')
def plan(cohort):
    models=[];extra=[Path(__file__)]
    for condition in ['silence','speech']:
        report=exporter.report(condition)
        if not report.exists():
            print(json.dumps(dict(status='not_ready',missing=str(report),model_calls=0)));return 2
        e=json.loads(report.read_text());assert e['all_static_and_runtime_abi_passed']
        model=exporter.output(condition)/'int8.tflite';assert exporter.exercised.sha(model)==e['model']['sha256']
        models.append(dict(name=f'context-{condition}',path=str(model),sha256=e['model']['sha256']))
        extra.extend([report,model,exporter.lock(condition)])
    module_name,original=SOURCES[cohort];module=importlib.import_module(module_name)
    p=json.loads(original.read_text());extra.extend([original,Path(module.__file__)])
    p['files'].update({str(f):exporter.exercised.sha(f) for f in extra})
    p.update(models=models,scope='Both final paired models on already consumed development inputs. Fixed .91796875/smoothing5/refractory67/half-threshold rearm/81600prime/native480. No new independent exposure, training, threshold selection or acceptance. Preserve all prime/gap/tail events where applicable. Container ownership is descriptive, not phonetic truth. Report both models, retain C161 baseline. Original source identity files retained unchanged.')
    with lock(cohort).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(status='planned',cohort=cohort)));return 0

def run(cohort):
    module=importlib.import_module(SOURCES[cohort][0])
    module.PLAN=lock(cohort);module.OUT=output(cohort)
    module.run()
if __name__=='__main__':
    action,cohort=sys.argv[1:];assert cohort in SOURCES and action in ['plan','run']
    raise SystemExit(plan(cohort) if action=='plan' else run(cohort))
