"""Complete old84/new171 training-fit census, never generalization evidence."""
import importlib.metadata
import json
from pathlib import Path
import sys
import replay_training_carrier_support_v1 as parent
import export_chatterbox_context_pair_v1 as exporter
import export_chatterbox_daily_context_v1 as daily_exporter

def lock(cohort):return parent.R/f'4090-daily-context-training-{cohort}-replay-plan-v1.json'
def output(cohort):return Path(f'.local/wakeword/daily-context-training-{cohort}-replay-v1')
def plan(cohort):
    er=exporter.report('daily-context')
    if not er.exists():print('not_ready: final daily-context export absent');return 2
    e=json.loads(er.read_text());assert e['all_static_and_runtime_abi_passed']
    model=Path(e['model']['path']);assert parent.sha(model)==e['model']['sha256']
    admission=parent.R/('4090-chatterbox-training-admission-v1.json' if cohort=='old84' else '4090-chatterbox-daily-training-admission-v1.json')
    rows=json.loads(admission.read_text())['admitted'];assert len(rows)==dict(old84=84,new171=171)[cohort]
    paths=[Path(__file__),Path(parent.__file__),Path(exporter.__file__),Path(daily_exporter.__file__),admission,er,exporter.lock('daily-context'),model]
    paths += [Path(__file__).with_name(n) for n in ['collect_multispeaker_formal_v1.py','uint8_event_policy_v1.py']]
    paths.append(Path('research/wakeword_p4/training/features.py'))
    for row in rows:
        wav=Path(row['wav']['path']);assert wav.stat().st_size==row['wav']['bytes'] and parent.sha(wav)==row['wav']['sha256'];paths.append(wav)
    p=dict(files={str(f):parent.sha(f) for f in paths},models={'daily-context':str(model)},rows=rows,
        versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},
        profile=dict(threshold=.91796875,smoothing=5,refractory=67,prime_samples=81600,native_return_samples=480),
        policy='Complete admitted training-only census, original full PCM, continuous zero prime, native480 return clocks, no boundary reset or tail extension. CPU1thread deployment simulation, not CPU training. No quarantine, acceptance, threshold or checkpoint selection. Targeted consumed training fit only; ASR agreement is provisional, not phonetic truth. No promotion or generalization claim.')
    with lock(cohort).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0
def run(cohort):
    parent.PLAN=lock(cohort);parent.OUT=output(cohort);parent.run()
if __name__=='__main__':
    action,cohort=sys.argv[1:];assert cohort in ['old84','new171']
    raise SystemExit({'plan':plan,'run':run}[action](cohort))
