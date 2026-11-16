"""Observe the original training, then perform one export and sequential fixed-profile replay."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments');GPU=Path('research/wakeword_p4/training/gpu_python.py');OUT=Path('.local/wakeword/c161-chatterbox-continuation-v1')
def run(command,stage):
    state=dict(stage=stage,pid=os.getpid(),time=time.time(),command=list(map(str,command)))
    (OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    with (OUT/(stage+'.log')).open('xb') as f:subprocess.run(list(map(str,command)),check=True,stdout=f,stderr=subprocess.STDOUT)
def publish(paths,message):
    run(['git','add',*paths],message+'-add')
    run(['git','commit','-m',message],message+'-commit')
    run(['git','push','origin','research-hye-nyx-voice-gpu'],message+'-push')
def main():
    OUT.mkdir(exist_ok=False)
    launch=json.loads(Path('.local/wakeword/c161-chatterbox-training-v1/launch.json').read_text());pid=launch['pid'];proc=Path('/proc')/str(pid)
    (OUT/'state.json').write_text(json.dumps(dict(stage='waiting-original-training',original=launch,pid=os.getpid()),indent=2)+'\n')
    while proc.exists():
        try:
            if proc.joinpath('stat').read_text().split()[21]!=launch['proc_starttime']:break
        except FileNotFoundError:break
        time.sleep(5)
    report=R/'4090-c161-chatterbox-training-original-v1.json';r=json.loads(report.read_text())
    assert r['steps']==r['optimizer_iterations']==20000 and r['all_training_checks_passed'] and r['plan_sha256']==launch['plan_sha256']
    py=sys.executable
    run([py,E/'export_c161_chatterbox_v1.py','plan'],'export-plan')
    publish([str(report),str(R/'4090-c161-chatterbox-int8-plan-v1.json')],'research-pitch-complete-training-export-plan')
    run([py,GPU,E/'export_c161_chatterbox_v1.py','run'],'export-run')
    export=R/'4090-c161-chatterbox-int8-original-v1.json';er=json.loads(export.read_text());assert er['all_static_and_runtime_abi_passed']
    publish([str(export)],'research-pitch-original-int8-export')
    for cohort in ['primed','jenny','joe']:
        collector=E/f'collect_c161_chatterbox_{cohort}_v1.py'
        run([py,collector,'plan'],cohort+'-plan')
        publish([str(R/f'4090-c161-chatterbox-{cohort}-plan-v1.json')],'research-pitch-'+cohort+'-execution')
        run([py,collector,'run'],cohort+'-collect')
        run([py,E/f'score_c161_chatterbox_{cohort}_v1.py'],cohort+'-score')
    for tag,script in [('mac12','replay_c161_chatterbox_mac_witness_v1.py'),('builtin36','replay_candidate_chatterbox_builtin_v1.py')]:
        run([py,E/script,'plan'],tag+'-plan')
        filename='4090-c161-chatterbox-mac12-plan-v1.json' if tag=='mac12' else '4090-c161-chatterbox-builtin-replay-lock-v1.json'
        publish([str(R/filename)],'research-chatterbox-'+tag+'-execution')
        run([py,E/script,'run'],tag+'-collect')
    (OUT/'state.json').write_text(json.dumps(dict(stage='completed-replay-results-await-review',pid=os.getpid(),time=time.time()),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.log').write_text(traceback.format_exc())
        raise
