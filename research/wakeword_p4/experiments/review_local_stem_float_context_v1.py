"""Independent saved float/INT8 event decoding and archive verification."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

R = Path('research/wakeword_p4/results')
source = R / '4090-local-stem-float-context-original-v1.json'
raw = source.read_bytes()
d = json.loads(raw)
archive = d['archive']
assert hashlib.sha256(Path(archive['path']).read_bytes()).hexdigest() == archive['sha256']
assert Path(archive['path']).stat().st_size == archive['bytes']
expected = {Path(row['trace']['path']).name: row['trace'] for row in d['rows']}
assert len(expected) == len(d['rows']) == 3
with tarfile.open(archive['path']) as t:
    assert set(t.getnames()) == set(expected) | {'records.jsonl'}
    assert len(t.getnames()) == 4
    ledger = [json.loads(s) for s in t.extractfile('records.jsonl').read().splitlines()]
    assert ledger == d['rows']
    for name, spec in expected.items():
        data = t.extractfile(name).read()
        assert len(data) == spec['bytes']
        assert hashlib.sha256(data).hexdigest() == spec['sha256']


def decode(values, threshold):
    cumulative = np.r_[0., np.cumsum(values, dtype=np.float64)]
    sums = cumulative[5:] - cumulative[:-5]
    armed, last, steps = True, -1000, []
    for step, total in enumerate(sums, 5):
        if step - last < 67: continue
        if total < threshold*2.5: armed = True
        if armed and total >= threshold*5:
            steps.append(step); last = step; armed = False
    return steps


rows = []
for row in d['rows']:
    trace = row['trace']
    assert hashlib.sha256(Path(trace['path']).read_bytes()).hexdigest() == trace['sha256']
    with np.load(trace['path'], allow_pickle=False) as z:
        clock = z['relative_block_return']
        probs = z['probability'].copy()
        assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        assert z['uint8'].dtype == np.uint8
        events = {kind: [dict(step=s, relative_sample=int(clock[s-1]))
                         for s in decode(values, threshold)]
                  for kind, values, threshold in [('float', probs, 235/256), ('int8', z['uint8'], 235)]}
    assert events == row['events']
    groups, clips = {}, []
    # Direct per-window means provide a separately calculated margin diagnostic.
    means = np.array([probs[i-5:i].astype(np.float64).mean() for i in range(5, len(probs)+1)])
    for item in row['clips']:
        clip = item['clip']
        hits = {kind: any(clip['start'] < e['relative_sample'] <= clip['end'] for e in ev)
                for kind, ev in events.items()}
        assert hits == item['hits']
        g = groups.setdefault(clip['group'], dict(clips=0, float_hits=0, int8_hits=0))
        g['clips'] += 1
        for kind in hits: g[kind+'_hits'] += hits[kind]
        selected = (clock[4:] > clip['start']) & (clock[4:] <= clip['end'])
        clips.append(dict(clip=clip, hits=hits, max_float_five_mean=float(means[selected].max())))
    rows.append(dict(arm=row['arm'], name=row['name'], groups=groups, clips=clips))
out = dict(schema='local-stem-float-context-review-v1', all_passed=True,
           source_sha256=hashlib.sha256(raw).hexdigest(), rows=rows, verified_archive_members=4,
           limits='Three consumed complete contexts only. Independent saved-output decoding; '
           'no new inference/exposure or phonetic attribution. Does not explain Joe outcomes or prove P4 fitness.')
with (R / '4090-local-stem-float-context-review-v1.json').open('x') as f:
    json.dump(out, f, indent=2); f.write('\n')
print(json.dumps([dict(arm=r['arm'], name=r['name'], groups=r['groups']) for r in rows], indent=2))
