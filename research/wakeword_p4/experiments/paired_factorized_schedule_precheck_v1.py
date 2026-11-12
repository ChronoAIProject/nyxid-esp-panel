"""Candidate-independent paired geometry precheck, never training/mining.

50 blocks, each two different numeric conditioning IDs at one common speed,
all four phrases/five arms, and one common absolute endpoint across all40 views.
Uniform draws use a new local RNG only; inherited training samplers untouched.
Phase-one geometry follows Mac19aed75 common_ends: raw[e-149:e+1], e%3=0,
e>=feature_crop_end and e<=min(350, raw_length-1). The graph's last frame is
unused, hence signal containment is checked against effective exclusive end e.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('.local/wakeword/factorized-training-features-v1')
digest = hashlib.sha256((root/'report.json').read_bytes()).hexdigest()
assert digest == '7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
records = json.loads((root/'report.json').read_text())['records']
phrases = ['hey_nyx', 'hey_nick', 'hey_next', 'hey_mix']
arms = ['quiet', 'noise25', 'combined', 'noise15', 'rir']
voices = sorted({int(r['voice']) for r in records})
speeds = [.9, 1., 1.1]
assert voices == sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
groups = {}
for r in records:
    key = (int(r['voice']), r['length_multiplier'], r['target'], r['arm'])
    assert key not in groups and r['split'] == 'training'
    groups[key] = r
assert len(groups) == 40320
def rows_for(ids, speed):
    return [groups[v, speed, p, a] for v in ids for p in phrases for a in arms]
def legal(rows):
    lo = max(r['feature_crop_end'] for r in rows)
    hi = min(350, *(r['raw_shape'][0]-1 for r in rows))
    ends = np.arange(((lo+2)//3)*3, hi+1, 3, dtype=np.int64)
    assert len(ends) and lo >= 150
    return ends

geometry = []
for v in voices:
    for s in speeds:
        rows = rows_for([v], s); ends = legal(rows)
        geometry.append(dict(voice=v, speed=s, first=int(ends[0]), last=int(ends[-1]), count=len(ends)))
# Certify every possible two-ID pair shares at least one legal endpoint:
# intersection of every individual interval contains this per-speed interval.
global_intersections = {}
for speed in speeds:
    ends = legal([r for r in records if r['length_multiplier'] == speed])
    global_intersections[str(speed)] = dict(first=int(ends[0]), last=int(ends[-1]), count=len(ends))
rng = np.random.default_rng(227726)
blocks = []
for block in range(50):
    speed = float(rng.choice(speeds)); ids = [int(v) for v in rng.choice(voices, size=2, replace=False)]
    rows = rows_for(ids, speed); ends = legal(rows); end = int(rng.choice(ends))
    assert len(rows) == 40 and ids[0] != ids[1]
    views = []
    for r in rows:
        # All raw hashes were independently audited before this metadata precheck.
        # Verify actual sampled files again, including exact window bytes.
        path = root/r['raw']; data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == r['raw_file_sha256']
        raw = np.load(path, allow_pickle=False)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == r['raw_sha256']
        x = raw[end-149:end+1]
        assert x.shape == (150, 40) and end%3 == 0
        assert (end-150)*160 <= 32000 and (end+1)*160 >= 32000+r['signal_samples']
        views.append(dict(pack_index=r['pack_index'], voice=r['voice'], phrase=r['target'], arm=r['arm'],
                          raw_file_sha256=r['raw_file_sha256'], window_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    blocks.append(dict(block=block, voices=ids, speed=speed, endpoint_inclusive=end, views=views))
result = dict(scope='Paired training-only sampling feasibility; no model inference or updates',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), report_sha256=digest,
    seed=227726, geometry=geometry, global_per_speed_intersections=global_intersections,
    blocks=blocks, checked_views=2000, all_672_ids_3_speeds_4_phrases_5_arms_present=True,
    limits='Potential pairing schedule only, not an adopted batch mixture/loss or training run. Conditioning IDs are synthetic, not independently recorded people. No new/final data, threshold search or performance evidence.')
Path('research/wakeword_p4/results/4090-paired-factorized-schedule-precheck-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(blocks=50, checked_views=2000, global_intersections=global_intersections)))
