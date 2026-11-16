"""Paired saved-event changes, no waveform inference or selection."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    inputs={};groups={}
    for cohort in ['primed','jenny']:
        paths=[Path(f'.local/wakeword/{prefix}-{cohort}-v1/diagnostic/hey_nyx-source-balanced.jsonl') for prefix in ['causal-source-balanced','c161-chatterbox']]
        data=[json.loads(p.read_text().splitlines()[0]) for p in paths]
        a,b=data;assert a['summary']['threshold']==b['summary']['threshold']==.91796875
        assert len(a['records'])==len(b['records'])
        changes=collections.Counter();cases=[];false_by_phrase=collections.Counter()
        for old,new in zip(a['records'],b['records'],strict=True):
            assert (old['phase'],old['index'],old['source_id'])==(new['phase'],new['index'],new['source_id'])
            if old['phase']=='background':continue
            assert (old['target_id'],old['variant'])==(new['target_id'],new['variant'])
            target=old['target_id']=='hey_nyx'
            oh=old['positive']['any_after_source'] if target else bool(old['events'])
            nh=new['positive']['any_after_source'] if target else bool(new['events'])
            label=('target/' if target else 'negative/')+('retained_hit' if oh and nh else 'lost' if oh else 'gained' if nh else 'retained_silent')
            changes[label]+=1
            if not target and nh:false_by_phrase[new['target_id']]+=1
            if oh!=nh:cases.append(dict(source_id=old['source_id'],variant=old['variant'],target_id=old['target_id'],old_activated=oh,new_activated=nh,carrier=old['carrier']))
        groups[cohort]=dict(baseline=a['summary'],candidate=b['summary'],paired_counts=dict(changes),candidate_false_by_phrase=dict(false_by_phrase),changed_cases=cases)
        inputs.update({str(p):sha(p) for p in paths})
    for name,old_path,new_path in [
        ('builtin36','.local/wakeword/c161-chatterbox-v1/result.json','.local/wakeword/c161-chatterbox-builtin-v1/result.json'),
        ('mac12','.local/wakeword/c161-mac12-v1/result.json','.local/wakeword/c161-chatterbox-mac12-v1/result.json')]:
        paths=[Path(old_path),Path(new_path)]
        a,b=[json.loads(path.read_text()) for path in paths]
        assert len(a['records'])==len(b['records'])
        changes=collections.Counter();cases=[]
        for old,new in zip(a['records'],b['records'],strict=True):
            assert old['index']==new['index']
            key='request' if name=='builtin36' else 'source'
            assert old[key]==new[key]
            oh=bool(old['real_events']);nh=bool(new['real_events'])
            group=old['request']['group'] if name=='builtin36' else str(old['source']['target'])
            change='retained_hit' if oh and nh else 'lost' if oh else 'gained' if nh else 'retained_silent'
            changes[group+'/'+change]+=1
            if oh!=nh:cases.append(dict(index=old['index'],source=old[key],old_activated=oh,new_activated=nh))
        groups[name]=dict(paired_counts=dict(changes),changed_cases=cases)
        inputs.update({str(path):sha(path) for path in paths})
    d=dict(schema='c161-chatterbox-paired-comparison-v1',inputs=inputs,groups=groups,
        decision='pending_evidence_review',
        policy='Summarize complete saved events at fixed original threshold only; no automatic candidate promotion or rejection. Constructed nearwords diagnostic, prioritize common-language conflicts and voice/command coverage. Reused backgrounds nonadditive. Human/P4/resource conclusions require separate evidence.')
    (R/'4090-c161-chatterbox-paired-comparison-v1.json').write_text(json.dumps(d,indent=2)+'\n')
    for group,value in groups.items():print(group,value['paired_counts'],value.get('candidate_false_by_phrase',{}))
if __name__=='__main__':main()
