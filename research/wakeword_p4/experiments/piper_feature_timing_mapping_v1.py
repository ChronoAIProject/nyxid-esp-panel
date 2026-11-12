"""Map current training feature rows to original saved Piper generation metadata."""
import hashlib
import json
from pathlib import Path
from collections import Counter

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
root = Path('.local/wakeword/features-gpu-ami-extra-v1')
manifest = Path('.local/wakeword/hey-nyx/piper-gpu/manifest.jsonl')
original = {}
for line in manifest.open():
    r = json.loads(line)
    if r['split'] == 'training' and r['label'] == 1:
        original['piper-gpu/'+r['path']] = r
assert len(original) == 6000
counts = Counter(); variants = Counter(); mapped = []; seen = set()
for i, r in enumerate(json.loads((root/'features.json').read_text())):
    if r['split'] != 'training' or r['label'] != 1 or r.get('engine', 'piper') != 'piper':
        continue
    source = original[r['path']]
    for key in ['sha256', 'samples', 'keyword_end', 'text', 'speakers', 'length_scale']:
        assert r[key] == source[key], (i, key)
    assert sha(root/'dataset'/r['path']) == source['sha256']
    assert 0 <= r['keyword_frame']-(r['lead_frame']+r['keyword_end']//160-2) <= 1
    counts[r['path']] += 1; variants[str(r['variant'])] += 1; seen.add(r['path'])
    mapped.append({'feature_index': i, 'source': r['path'], 'sha256': r['sha256'], 'variant': r['variant'], 'lead_frame': r['lead_frame'], 'keyword_frame': r['keyword_frame'], 'frames': r['frames'], 'offset': r['offset'], 'generation_keyword_end': r['keyword_end']})
assert len(mapped) == 60000 and seen == set(original) and set(counts.values()) == {10}
out = {
    'scope': 'Actual main sampler Piper positive training rows mapped to original generation timing.',
    'inputs': {str(p): sha(p) for p in [root/'features.json', manifest]},
    'mapped_training_feature_rows': len(mapped), 'unique_original_sources': len(seen),
    'rows_per_source': 10, 'variant_counts': dict(variants), 'rows': mapped,
    'limits': 'All referenced WAV bytes authenticated. Feature binary bytes not audited here. lead_frame loses subframe sample offset; no exact lead sample claim. RIR tail and feature/context support predicate still need explicit derivation before endpoint substitution. Generation timing is synthetic reference, not actual phone truth. No training, inference, development/final waveform access or promotion.',
}
Path('research/wakeword_p4/results/4090-piper-feature-timing-mapping-v1.json').write_text(json.dumps(out, separators=(',', ':'))+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['rows','inputs','limits','scope']}))
