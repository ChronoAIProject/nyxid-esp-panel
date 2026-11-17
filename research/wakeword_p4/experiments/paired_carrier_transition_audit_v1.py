"""Describe frozen consumed outcomes; no inference, tuning, or causal attribution."""
import hashlib
import json
from pathlib import Path

R = Path('research/wakeword_p4/results')
inputs = {}


def read(name):
    path = R / name
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def counter():
    return dict(clips=0, control_hits=0, paired_hits=0, gained=0, lost=0)


def add(counts, before, after):
    counts['clips'] += 1
    counts['control_hits'] += before
    counts['paired_hits'] += after
    counts['gained'] += after and not before
    counts['lost'] += before and not after


review = read('4090-binary40-paired-carrier-consumed-review-v1.json')
assert review['all_passed']
cohorts = {}
for cohort in ['primed', 'jenny', 'joe', 'melo']:
    arms = {a: read(f'4090-binary40-paired-carrier-{a}-{cohort}-comparison-v1.json')
            for a in ['control', 'paired']}
    groups, changes = {}, []
    assert len(arms['control']['records']) == len(arms['paired']['records'])
    for before, after in zip(arms['control']['records'], arms['paired']['records']):
        assert before['source'] == after['source'] and before['index'] == after['index']
        assert before['phase'] == after['phase']
        if before['hits'] is None:
            continue
        source = before['source']
        target = source.get('phrase_id') == 'nyx' if cohort == 'melo' else source['target_id'] == 'hey_nyx'
        voice = source.get('voice_id', cohort)
        carrier = source.get('carrier', 'unclassified-whole-output')
        group = f'{voice}/{"target" if target else "negative"}/{carrier}'
        b, a = before['hits']['binary40'], after['hits']['binary40']
        add(groups.setdefault(group, counter()), b, a)
        if b != a:
            changes.append(dict(index=before['index'], target=target, group=group,
                                control_hit=b, paired_hit=a, source=source,
                                control_events=before['events']['binary40'],
                                paired_events=after['events']['binary40']))
    cohorts[cohort] = dict(groups=groups, changes=changes)

contexts = {}
arms = {a: read(f'4090-binary40-paired-carrier-{a}-context-original-v1.json')
        for a in ['control', 'paired']}
assert len(arms['control']['streams']) == len(arms['paired']['streams']) == 3
for before, after in zip(arms['control']['streams'], arms['paired']['streams']):
    assert before['name'] == after['name'] and before['timeline'] == after['timeline']
    groups, changes = {}, []
    for clip in before['timeline']:
        if clip['kind'] != 'clip':
            continue
        hits = [any(clip['start'] < e['relative_sample'] <= clip['end'] for e in row['events'])
                for row in [before, after]]
        add(groups.setdefault(clip['group'], counter()), *hits)
        if hits[0] != hits[1]:
            changes.append(dict(clip=clip, control_hit=hits[0], paired_hit=hits[1]))
    contexts[before['name']] = dict(groups=groups, changes=changes)

out = dict(schema='paired-carrier-transition-audit-v1', inputs=inputs,
           cohorts=cohorts, contexts=contexts,
           limits='Saved consumed outcomes only. No new inference/exposure or causal attribution. '
                  'Variant rows are correlated; whole-output timings and context containers '
                  'do not identify phonetic cause or first-command retention. No tuning or promotion.')
dest = R / '4090-binary40-paired-carrier-transition-audit-v1.json'
with dest.open('x') as f:
    json.dump(out, f, indent=2)
    f.write('\n')
print(json.dumps({c: v['groups'] for c, v in contexts.items()}, indent=2))
