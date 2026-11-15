"""Summarize native full-wave app events; preserve finish events explicitly."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from hey_development_metrics_v1 import far_upper
from sherpa_open_vocabulary_reference_v1 import OUT,LOCK,R,sha

def main():
    path=OUT/'report.json';original=json.loads(path.read_text())
    assert original['completed'] and len(original['records'])==408 and original['plan_sha256']==sha(LOCK)
    positives=[];neg=collections.defaultdict(lambda:dict(clips=0,activated_clips=0,events=0,finish_events=0));bg_events=0;bg_finish=0;hours=0.;prime=0;details=[]
    for rec in original['records']:
        row=rec['record'];events=rec['events'];prime+=sum(e['prime'] for e in events)
        real=[e for e in events if not e['prime']]
        details.append(dict(phase=rec['phase'],index=rec['index'],source_id=row['source_id'],events=events))
        if rec['phase']=='background':
            hours+=row['samples']/16000/3600;bg_events+=len(real);bg_finish+=sum(e['finish'] for e in real);continue
        if row['target_id']=='hey_nyx':
            after=[e for e in real if e['return_relative_sample']>=row['lead_samples']]
            in_source=[e for e in after if not e['finish']]
            positives.append(dict(source_id=row['source_id'],variant=row['variant'],carrier=row['carrier'],length_scale=row['length_scale'],hit_including_native_finish=bool(after),hit_before_source_input_finished=bool(in_source),native_finish_only=bool(after) and not in_source,first_signed_generated_prefix_ms=(after[0]['return_relative_sample']-row['generated_prefix_end_sample'])/16 if after else None,events=after))
        else:
            kind='cross_target' if row['target_id'] in ['hi_esp','hey_willow'] else 'common'
            n=neg[kind];n['clips']+=1;n['activated_clips']+=bool(real);n['events']+=len(real);n['finish_events']+=sum(e['finish'] for e in real)
    def summarize(rows):
        lat=[r['first_signed_generated_prefix_ms'] for r in rows if r['hit_including_native_finish']]
        return dict(clips=len(rows),hits_including_native_finish=sum(r['hit_including_native_finish'] for r in rows),hits_before_source_input_finished=sum(r['hit_before_source_input_finished'] for r in rows),native_finish_only=sum(r['native_finish_only'] for r in rows),first_signed_generated_prefix_median_ms=float(np.median(lat)) if lat else None)
    assert len(positives)==36 and neg['common']['clips']==288 and neg['cross_target']['clips']==72
    summary=dict(schema='sherpa-open-vocabulary-reference-behavior-v1',original_report_sha256=sha(path),execution_sha256=sha(LOCK),positive=dict(all=summarize(positives),**{c:summarize([r for r in positives if r['carrier']==c]) for c in sorted({r['carrier'] for r in positives})}),negative=dict(neg),background=dict(events_including_finish=bg_events,finish_events=bg_finish,hours=hours,far_upper95=far_upper(bg_events,hours)),priming_events_ignored_by_app=prime,elapsed_seconds=original['elapsed_seconds'],threshold_selected=False,candidate_promoted=False,limits=original['policy'])
    with (R/'4090-sherpa-open-vocabulary-reference-behavior-v1.json').open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
    # Preserve original report and all event-log bytes, without copied PCM/weights.
    stored=[]
    for source,label in [(path,'report'),(OUT/'events.jsonl','events')]:
        data=source.read_bytes();packed=gzip.compress(data,mtime=0);dest=R/f'4090-sherpa-open-vocabulary-reference-original-{label}-v1{source.suffix}.gz'
        with dest.open('xb') as f:f.write(packed)
        assert gzip.decompress(dest.read_bytes())==data
        stored.append(dict(path=str(dest),bytes=len(packed),sha256=sha(dest),original_bytes=len(data),original_sha256=hashlib.sha256(data).hexdigest()))
    receipt=dict(schema='sherpa-open-vocabulary-reference-evidence-v1',files=stored,original_records=408,all_original_bytes_preserved=True,published_bytes=sum(r['bytes'] for r in stored),threshold_selected=False,candidate_promoted=False)
    with (R/'4090-sherpa-open-vocabulary-reference-evidence-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(summary));print(json.dumps(receipt))
if __name__=='__main__':main()
