"""Paired saved application events; no inference, threshold search or audio access."""
import collections
import hashlib
import json
from pathlib import Path

R = Path('research/wakeword_p4/results')

def main():
    inputs, groups = [], []
    for group in ('primed', 'jenny'):
        rows = []
        for tag in ('source-balanced', 'negative-peak'):
            p = Path(f'.local/wakeword/causal-{tag}-{group}-v1/diagnostic/hey_nyx-{tag}.jsonl')
            raw = p.read_bytes()
            inputs.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest()))
            rows.append(json.loads(raw)['records'])
        assert len(rows[0]) == len(rows[1])
        cases, counts = [], collections.Counter()
        for old, new in zip(*rows):
            keys = ('phase', 'index', 'source_id', 'target_id', 'variant', 'carrier', 'length_scale')
            assert all(old.get(k) == new.get(k) for k in keys)
            if old['phase'] != 'synthetic':
                continue
            target = old['target_id'] == 'hey_nyx'
            before, after = (bool(x['positive']['any_after_source']) if target else bool(x['events']) for x in (old, new))
            status = ('retained_hit' if before else 'retained_miss') if target else ('retained_false' if before else 'retained_silent')
            if before != after:
                status = ('lost_hit' if before else 'gained_hit') if target else ('removed_false' if before else 'new_false')
            counts[status] += 1
            if before != after or (not target and after):
                cases.append(dict(**{k:old.get(k) for k in keys}, change=status, old_events=old['events'], new_events=new['events']))
        groups.append(dict(group=group, counts=dict(counts), cases=cases))
    out = dict(schema='causal-negative-peak-event-delta-v1', inputs=inputs, groups=groups,
               limits='Consumed development saved events at fixed threshold only. Correlated variants, no independent acceptance, phonetic inference, model calls, new audio, threshold rescue or training. Changes describe behavior, not causal explanation.')
    with (R/'4090-causal-negative-peak-event-delta-v1.json').open('x') as f:
        f.write(json.dumps(out, indent=2)+'\n')
    for group in groups:
        print(json.dumps(dict(group=group['group'], counts=group['counts'])))
        print(json.dumps([dict(source_id=x['source_id'],target=x['target_id'],carrier=x['carrier'],variant=x['variant'],speed=x['length_scale'],change=x['change']) for x in group['cases'] if x['change'] in ('lost_hit','gained_hit','new_false')]))

if __name__ == '__main__':
    main()
