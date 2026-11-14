"""Read saved training schedules and events; no model or audio inference."""
import gzip
import hashlib
import json
from pathlib import Path

R=Path('research/wakeword_p4/results')
BASE=Path('.local/wakeword/multispeaker-formal-v1')
F=Path('.local/wakeword/multispeaker-coverage-features-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    lock_path=R/'4090-multispeaker-formal-execution-v1.json'
    plan=json.loads(lock_path.read_text())
    report_path=R/'4090-multispeaker-formal-original-v1.json'
    report=json.loads(report_path.read_text());assert report['execution_sha256']==sha(lock_path)
    rows=[]
    for arm in ['narrow','broad']:
        metadata=F/arm/'features.json';assert sha(metadata)==plan['files'][str(metadata)]
        by_id={r['shared_index']:r for r in json.loads(metadata.read_text())}
        schedule=BASE/arm/'ordered-batches.jsonl.gz'
        actual=next(a for a in report['arms'] if a['arm']==arm)
        assert sha(schedule)==actual['files']['ordered-batches.jsonl.gz']
        counts={};steps=0
        with gzip.open(schedule,'rt') as stream:
            for line in stream:
                receipt=json.loads(line);steps+=1;assert receipt['step']==steps
                for d in receipt['descriptors']:
                    key=d['kind']+'/'+str(d['label'])
                    s=counts.setdefault(key,dict(windows=0,min_end=None,max_end=None,end_before_309=0,synthetic_source_not_started=0))
                    end=d['end'];s['windows']+=1
                    s['min_end']=end if s['min_end'] is None else min(s['min_end'],end)
                    s['max_end']=end if s['max_end'] is None else max(s['max_end'],end)
                    s['end_before_309']+=int(end<309)
                    if d['kind']=='synthetic':
                        row=by_id[d['index']]
                        s['synthetic_source_not_started']+=int(end<=row['lead_frame'])
        assert steps==20000 and sum(s['windows'] for s in counts.values())==2560000
        rows.append(dict(arm=arm,steps=steps,schedule_sha256=sha(schedule),counts=counts,
                         synthetic_lead_frames=sorted({r['lead_frame'] for r in by_id.values()})))
    historical=[Path(__file__).with_name(n) for n in ['phase_aligned_raw_findings_v1.md','hey_broad_consistency_protocol_v1.md','hey_broad_paired_qat_protocol_v1.md']]
    event_path=R/'4090-multispeaker-formal-saved-pre-source-event-summary-v1.json'
    events=json.loads(event_path.read_text());endpoints=[]
    for arm in ['narrow','broad']:
        row=next(p for p in events['points'] if p['arm']==arm and p['threshold']==.99609375)
        endpoints.append(row)
    output=dict(schema='multispeaker-startup-coverage-review-v1',files={str(p):sha(p) for p in [Path(__file__),lock_path,report_path,event_path,*historical]},arms=rows,
                fixed_max_threshold_saved_events=endpoints,model_invocations=0,optimizer_updates=0,
                observations=['All actual training inputs were complete309-frame windows, with no endpoint before frame309. This is not exposure to partial-startup streaming states.',
                'Synthetic source lead was fixed310frontend frames; some negative windows can precede source placement, but these remain full309-frame contexts.',
                'Saved maximum-grid-threshold events precede source placement at .44s narrow and .74–.86s broad, substantially before309frame context is available.',
                'Reviewed phase-alignment changes only nominal versus shifted full-window supervision; consistency/QAT protocols target quantization agreement/export initialization. They do not describe fullsequence startup-negative supervision.'],
                limits='Bounded review of listed historical documents/current actual schedules/saved events only, not exhaustive proof of absence in all history or causal attribution. No new float comparisons, audio/model inference, threshold selection or freshdata. Startup-aligned sequence training remains a hypothesis, not an implemented or successful candidate.')
    with (R/'4090-multispeaker-startup-coverage-review-v1.json').open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))
if __name__=='__main__':main()
