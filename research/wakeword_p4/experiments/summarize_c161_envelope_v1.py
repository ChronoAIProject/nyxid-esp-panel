"""Full fixed-profile decision table from completed original reports, no inference."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    paths=[R/'4090-c161-envelope-continuation-original-v1.json',R/'4090-c161-envelope-training-original-v1.json',R/'4090-context-envelope-int8-original-v1.json',R/'4090-context-envelope-initial-comparison-v1.json',R/'4090-context-envelope-mac-comparison-v1.json',Path('.local/wakeword/c161-mac12-v1/result.json')]
    execution=json.loads(paths[0].read_text());assert execution['status']=='completed-await-review' and len(execution['stages'])==49 and all(s['actual_exit_code']==0 for s in execution['stages'])
    training=json.loads(paths[1].read_text());assert training['steps']==training['optimizer_iterations']==20000 and training['all_training_checks_passed']
    export=json.loads(paths[2].read_text());assert export['all_static_and_runtime_abi_passed'] and sha(export['model']['path'])==export['model']['sha256']
    cohorts={};archives=[]
    initial=json.loads(paths[3].read_text());archives.append(initial['archive'])
    for cohort in ['primed','jenny','joe','melo']:
        path=R/f'4090-context-envelope-{cohort}-comparison-v1.json';paths.append(path);report=json.loads(path.read_text());archives.append(report['archive']);table={}
        for key,g in report['groups'].items():
            is_target=key.startswith('target/') or '/target/' in key
            scope=key.split('/')[0] if cohort=='melo' else cohort
            record=table.setdefault(scope,dict(target_clips=0,bare_clips=0,command_clips=0,unclassified_clips=0,other_clips=0,models={n:dict(target=0,bare=0,command=0,unclassified=0,other=0) for n in g['hits']},target_gains=0,target_losses=0,other_added=0,other_removed=0))
            if is_target:
                kind='unclassified' if key.endswith('whole-utterance-unclassified') else 'bare' if key.endswith('target/') or key.endswith('target/bare') else 'command'
                record['target_clips']+=g['clips'];record[kind+'_clips']+=g['clips']
                record['target_gains']+=g['gained'];record['target_losses']+=g['lost']
                for name,hits in g['hits'].items():record['models'][name]['target']+=hits;record['models'][name][kind]+=hits
            else:
                record['other_clips']+=g['clips'];record['other_added']+=g['gained'];record['other_removed']+=g['lost']
                for name,hits in g['hits'].items():record['models'][name]['other']+=hits
        cohorts[cohort]=dict(table=table,reused_background=report['reused_background'])
    mac=json.loads(paths[4].read_text());archives.append(mac['archive'])
    for archive in archives:assert sha(archive['path'])==archive['sha256'] and Path(archive['path']).stat().st_size==archive['bytes']
    baseline_builtin=Path('.local/wakeword/c161-chatterbox-v1/result.json');baseline_lock=R/'4090-c161-chatterbox-replay-lock-v1.json';baseline_continuous=R/'4090-continuous-chatterbox-original-v1.json';baseline_daily=R/'4090-daily-language-pair-result-v1.json';paths.extend([baseline_builtin,baseline_lock,baseline_continuous,baseline_daily,Path(__file__)])
    builtin=json.loads(baseline_builtin.read_text());continuous=json.loads(baseline_continuous.read_text());daily=json.loads(baseline_daily.read_text())
    assert builtin['lock_sha256']==sha(baseline_lock)
    assert json.loads(baseline_lock.read_text())['files']['.local/wakeword/causal-source-balanced-int8-v1/int8.tflite']=='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
    mac_baseline=json.loads(Path('.local/wakeword/c161-mac12-v1/result.json').read_text())
    summary=dict(schema='c161-envelope-final-summary-v1',decision='retain-c161-do-not-promote-envelope',execution_stages=49,training_seconds=training['elapsed_seconds'],model=export['model'],source_files={str(p):sha(p) for p in paths},cohorts=cohorts,initial=initial['summary'],baseline_builtin=builtin['summary'],baseline_daily=daily['groups'],baseline_continuous=[dict(gap_samples=r['gap_samples'],pcm_sha256=r['pcm_sha256'],target_containers=len(r['models']['c161']['clip_containers_with_events']['nyx'])) for r in continuous['results']],mac=mac['groups'],mac_baseline_target_hits=sum(bool(r['real_events']) for r in mac_baseline['records'] if r['source']['target']=='hey_nyx'),archive_identities=[dict(path=a['path'],bytes=a['bytes'],sha256=a['sha256'],members=len(a['members'])) for a in archives],limits='Consumed correlated synthetic diagnostics, no new independent acceptance or additive background hours. Joe carrier unclassified. Other-synthetic totals include reciprocal targets, not daily rates. No threshold/calibration/checkpoint rescue; Mac host replay not P4. Broad research goal remains incomplete.')
    with (R/'4090-c161-envelope-final-summary-v1.json').open('x') as f:json.dump(summary,f,indent=2)
    print(json.dumps(dict(decision=summary['decision'],cohorts=cohorts),indent=2))
if __name__=='__main__':main()
