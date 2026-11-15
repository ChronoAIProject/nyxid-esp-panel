"""Compare saved consumed-development events across fixed interventions; no inference."""
import collections
import hashlib
import json
from pathlib import Path

TAGS=('source-balanced','character-auxiliary','phone-auxiliary','phrase-auxiliary','nonlinear-readout','bn-affine')
KEYS=('phase','index','source_id','target_id','variant','carrier','length_scale')
R=Path('research/wakeword_p4/results')

def main():
    inputs=[];groups=[]
    for group in ('primed','jenny'):
        runs=[]
        for tag in TAGS:
            path=Path(f'.local/wakeword/causal-{tag}-{group}-v1/diagnostic/hey_nyx-{tag}.jsonl')
            raw=path.read_bytes();data=json.loads(raw)
            assert data['summary']['threshold']==.91796875
            inputs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            runs.append(data['records'])
        assert len({len(x) for x in runs})==1
        cases=[];hist={'target_misses':collections.Counter(),'negative_activations':collections.Counter()}
        for records in zip(*runs,strict=True):
            first=records[0]
            assert all(all(first.get(k)==rec.get(k) for k in KEYS) for rec in records)
            if first['phase']!='synthetic':continue
            target=first['target_id']=='hey_nyx'
            active=[bool(x['positive']['any_after_source']) if target else bool(x['events']) for x in records]
            errors=[not x if target else x for x in active]
            hist['target_misses' if target else 'negative_activations'][sum(errors)]+=1
            if any(errors):
                cases.append(dict(**{k:first.get(k) for k in KEYS},target=target,errors=dict(zip(TAGS,errors,strict=True)),error_count=sum(errors)))
        persistent=[x for x in cases if x['error_count']==len(TAGS)]
        groups.append(dict(group=group,error_frequency_histogram={k:dict(v) for k,v in hist.items()},persistent_target_misses=sum(x['target'] for x in persistent),persistent_false_clips=sum(not x['target'] for x in persistent),persistent_false_by_phrase=dict(collections.Counter(x['target_id'] for x in persistent if not x['target'])),cases=cases))
    result=dict(schema='causal-intervention-persistence-v1',conditions=TAGS,inputs=inputs,groups=groups,limits='Saved consumed-development events at one fixed reference threshold only; no model calls, audio generation/access, threshold selection, causal attribution or training-data admission. Variants correlated. Joe complete misses already established separately; not included because its record schema differs. Frozenv30 retained; independent qualification incomplete.')
    with (R/'4090-causal-intervention-persistence-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    for group in groups:print(json.dumps({k:v for k,v in group.items() if k!='cases'}))

if __name__=='__main__':main()
