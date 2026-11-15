"""Describe existing actual wake events by full-wave context; no new model calls."""
import collections
import hashlib
import json
from pathlib import Path
R=Path('research/wakeword_p4/results')
NEAR={'hey_nick','hey_next','hey_mix','hey_muse'}
SOURCES={'95ab':'.local/wakeword/causal-multivoice-jenny-v1/diagnostic/hey_nyx-multivoice.jsonl','5557':'.local/wakeword/causal-positive-coverage-jenny-v1/diagnostic/hey_nyx-positive-coverage.jsonl','c161':'.local/wakeword/causal-source-balanced-jenny-v1/diagnostic/hey_nyx-source-balanced.jsonl'}
def main():
    reports=[];indices={};sources=[]
    for model,path in SOURCES.items():
        p=Path(path);x=json.loads(p.read_text().splitlines()[0]);rows=[r for r in x['records'] if r.get('target_id') in NEAR];assert len(rows)==144
        bykey={(r['source_id'],r['variant']):r for r in rows};assert len(bykey)==144;indices[model]=bykey
        active=[r for r in rows if r['events']];groups={}
        for field in ['target_id','carrier','variant','length_scale']:
            groups[field]=[dict(value=v,clips=sum(r[field]==v for r in rows),false_clips=sum(r[field]==v for r in active)) for v in sorted({r[field] for r in rows})]
        timing=collections.Counter()
        for r in active:
            e=r['events'][0]
            timing['before_generated_prefix' if e['generated_prefix_end_sample_signed_ms']<0 else 'at_or_after_generated_container' if e['generated_container_end_sample_signed_ms']>=0 else 'between_generated_prefix_and_container']+=1
        positive=[r for r in x['records'] if r.get('target_id')=='hey_nyx']
        reports.append(dict(model=model,nearword_clips=len(rows),false_clips=len(active),false_native_source_groups=len({r['source_id'] for r in active}),groups=groups,first_event_generated_coordinate_bins=dict(timing),target_hits_by_carrier={v:sum(r['positive']['any_after_source'] for r in positive if r['carrier']==v) for v in sorted({r['carrier'] for r in positive})},target_misses=[dict(source_id=r['source_id'],carrier=r['carrier'],variant=r['variant'],length_scale=r['length_scale']) for r in positive if not r['positive']['any_after_source']]))
        sources.append(dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pairs=[]
    for previous in ['95ab','5557']:
        assert set(indices[previous])==set(indices['c161'])
        counts=collections.Counter();changes=[]
        for key,r in indices['c161'].items():
            old=indices[previous][key]
            assert all(old[k]==r[k] for k in ['target_id','carrier','variant','length_scale'])
            a,b=bool(old['events']),bool(r['events']);kind='retained_false' if a and b else 'removed_false' if a else 'new_false' if b else 'both_silent';counts[kind]+=1
            if a!=b:changes.append(dict(source_id=key[0],variant=key[1],target_id=r['target_id'],carrier=r['carrier'],length_scale=r['length_scale'],change=kind))
        pairs.append(dict(previous=previous,current='c161',counts=dict(counts),changes=changes))
    out=dict(schema='causal-nearword-event-context-v1',sources=sources,models=reports,paired_transitions=pairs,threshold_selected=False,model_calls=0,limits='Existing consumed Jenny app-event records only; no new waveform, training, model inference, numerical agreement audit, threshold search or final access. Nearword subset144 excludes144 other common clips, which are silent. Three variants per source correlate; timing bins use generation coordinates, not actual phonemes, and early events are not automatic failure. Counts can reject exclusive explanations, not establish causal mechanism.')
    with (R/'4090-causal-nearword-event-context-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    for r in reports:print(json.dumps({k:v for k,v in r.items() if k not in ['target_misses','groups']}))
    for p in pairs:print(json.dumps(dict(previous=p['previous'],counts=p['counts'])))
if __name__=='__main__':main()
