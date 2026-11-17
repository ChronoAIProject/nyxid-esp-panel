"""Complete-only Binary40 comparison table, no automatic promotion or inference."""
import json
from pathlib import Path
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results')
def read(p):return json.loads(Path(p).read_text())
def main():
    sources=[]
    def load(p):sources.append(Path(p));return read(p)
    run=load(R/'4090-binary40-continuation-original-v1.json')
    assert run['status']=='completed-await-review' and len(run['stages'])==25 and all(s['actual_exit_code']==0 for s in run['stages'])
    terminal=load(R/'4090-binary40-training-terminal-v1.json');assert terminal['actual_exit_code']==0 and terminal['steps']==20000
    training=load(R/'4090-binary40-training-original-v1.json');assert terminal['report_sha256']==sha(sources[-1])
    export=load(R/'4090-binary40-trained-export-original-v1.json');assert export['training_report_sha256']==terminal['report_sha256']
    assert export['model']['sha256']==sha(export['model']['path']) and export['fresh_reset_exact'] and export['all_neural_tensors_integer']
    cohorts={};archives=[]
    for c in ['primed','jenny','joe','melo']:
        report=load(R/f'4090-binary40-{c}-comparison-v1.json');archives.append(report['archive']);table={}
        for key,g in report['groups'].items():
            target=key.startswith('target/') or '/target/' in key
            scope=key.split('/')[0] if c=='melo' else c
            row=table.setdefault(scope,dict(target_clips=0,bare_clips=0,command_clips=0,unclassified_clips=0,other_clips=0,models={n:dict(target=0,bare=0,command=0,unclassified=0,other=0) for n in g['hits']},target_gained=0,target_lost=0,other_added=0,other_removed=0))
            if target:
                kind='unclassified' if key.endswith('whole-utterance-unclassified') else 'bare' if key.endswith('target/') or key.endswith('target/bare') else 'command'
                row['target_clips']+=g['clips'];row[kind+'_clips']+=g['clips'];row['target_gained']+=g['gained'];row['target_lost']+=g['lost']
                for name,count in g['hits'].items():row['models'][name]['target']+=count;row['models'][name][kind]+=count
            else:
                row['other_clips']+=g['clips'];row['other_added']+=g['gained'];row['other_removed']+=g['lost']
                for name,count in g['hits'].items():row['models'][name]['other']+=count
        cohorts[c]=dict(table=table,reused_background=report['reused_background'])
    initial={}
    for c in ['builtin','daily','continuous','mac']:
        p=load(R/f'4090-binary40-{c}-comparison-v1.json');initial[c]=dict(groups=p['groups'],records=p['records']);archives.append(p['archive'])
    for a in archives:assert sha(a['path'])==a['sha256'] and Path(a['path']).stat().st_size==a['bytes']
    baseline_lock=R/'4090-c161-chatterbox-replay-lock-v1.json';bl=load(baseline_lock)
    model=Path('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite')
    identity='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
    assert bl['files'][str(model)]==sha(model)==identity
    builtin=load(Path('.local/wakeword/c161-chatterbox-v1/result.json'));assert builtin['lock_sha256']==sha(baseline_lock)
    continuous=load(R/'4090-continuous-chatterbox-original-v1.json')
    daily=load(R/'4090-daily-language-pair-result-v1.json')
    mac=load(Path('.local/wakeword/c161-mac12-v1/result.json'))
    # Same two continuous PCM identities, with no invented new exposure.
    for row in initial['continuous']['records']:
        original,=[r for r in continuous['results'] if r['gap_samples']==row['gap_samples']]
        assert original['pcm_sha256']==row['pcm_sha256']
    result=dict(schema='binary40-final-summary-v1',review_status='complete-evidence-awaits-human-agent-review-no-promotion',
                model=export['model'],training_seconds=training['elapsed_seconds'],execution_stages=25,
                cohorts=cohorts,initial=initial,baseline_builtin=builtin['summary'],baseline_daily=daily['groups'],
                baseline_continuous=[dict(gap_samples=r['gap_samples'],target_containers=len(r['models']['c161']['clip_containers_with_events']['nyx']),pcm_sha256=r['pcm_sha256']) for r in continuous['results']],
                baseline_mac_target_hits=sum(bool(r['real_events']) for r in mac['records'] if r['source']['target']=='hey_nyx'),
                archives=[{k:a[k] for k in ['path','bytes','sha256']} for a in archives],source_files={str(p):sha(p) for p in [*sources,Path(__file__)]},
                limits='Consumed correlated synthetic diagnostics; original C161 remains baseline. New architecture/init/optimizer combined intervention. No background hours summed across candidates; other-synthetic counts are not daily rates. No invented Joe carrier or phonetic endpoints, no human/P4/first-command retention qualification, threshold/checkpoint/calibration rescue or independent acceptance access.')
    with (R/'4090-binary40-final-summary-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(cohorts=cohorts,model=result['model']),indent=2))
if __name__=='__main__':main()
