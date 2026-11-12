"""Census authenticated completed training schedule; no waveform/model scoring."""
import collections
import hashlib
import json
from pathlib import Path
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
root = Path('.local/wakeword/model-matched-noise-independent-formal-v1')
p = root/'paired-sampling-trace.jsonl'
t = json.loads((root/'training.json').read_text())
assert sha(p) == t['paired_rendering']['sampling_trace_sha256']
counts = collections.Counter()
n = 0
for line in p.open():
    row = json.loads(line)
    assert row['block'] == n
    for voice in row['voices']:
        counts[(voice,row['speed'])] += 1
    n += 1
assert n == 120000
report = dict(steps=n, source_group_draws=sum(counts.values()),
    unique_voice_speed_groups=len(counts),
    draw_count_histogram=dict(sorted(collections.Counter(counts.values()).items())),
    repeated_group_draws_beyond_first=sum(v-1 for v in counts.values()),
    trace_sha256=sha(p), renderer_sha256=sha(Path('research/wakeword_p4/training/matched_noise_sampler.py')),
    noise_seed_scope='voice,speed,phrase only; cached across steps',
    limits='Counts certify reuse of source-group identities under deterministic seed formula, not memorization causality, quality, or a benefit of fresh noise. No new model scoring or development/final data.')
# Reproduce the already preserved receipt without rewriting it.
p = Path('research/wakeword_p4/results/4090-per-draw-noise-reuse-census-v1.json')
assert json.loads(json.dumps(report)) == json.loads(p.read_text())
print('Authenticated schedule census exactly reproduces preserved receipt')
