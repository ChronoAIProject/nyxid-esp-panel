"""Single daily-context model, all consumed builtin/daily/continuous inputs unchanged."""
import importlib,json
from pathlib import Path
import sys
import replay_context_pair_v1 as parent
import export_chatterbox_context_pair_v1 as exporter
import export_chatterbox_daily_context_v1 as daily_exporter
R=parent.R

def lock(cohort):return R/f'4090-context-daily-context-{cohort}-replay-plan-v1.json'
def output(cohort):return Path(f'.local/wakeword/context-daily-context-{cohort}-replay-v1')
def plan(cohort):
    report=exporter.report('daily-context')
    if not report.exists():print('not_ready: final daily-context export absent');return 2
    e=json.loads(report.read_text());assert e['all_static_and_runtime_abi_passed']
    model=exporter.output('daily-context')/'int8.tflite';assert exporter.exercised.sha(model)==e['model']['sha256']
    module_name,original=parent.SOURCES[cohort];module=importlib.import_module(module_name)
    p=json.loads(original.read_text())
    paths=[Path(daily_exporter.__file__),Path(exporter.__file__),Path(__file__),Path(parent.__file__),Path(module.__file__),original,report,model,exporter.lock('daily-context')]
    p['files'].update({str(f):exporter.exercised.sha(f) for f in paths})
    p.update(models=[dict(name='context-daily-context',path=str(model),sha256=e['model']['sha256'])],
        scope='One fixedfinal daily-context-context model on all consumed builtin36/daily96/continuous sessions as selected by cohort. Original PCM and native480/81600continuous prime/event profile retained. No threshold search, fitting, new exposure or acceptance. Preserve all events and traces; full-container events are not phoneme/first-command proof. Saved C161 and mixed arms remain comparison references, no automatic promotion.')
    with lock(cohort).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run(cohort):
    module=importlib.import_module(parent.SOURCES[cohort][0]);module.PLAN=lock(cohort);module.OUT=output(cohort);module.run()
if __name__=='__main__':
    action,cohort=sys.argv[1:];assert cohort in parent.SOURCES
    raise SystemExit({'plan':plan,'run':run}[action](cohort))
