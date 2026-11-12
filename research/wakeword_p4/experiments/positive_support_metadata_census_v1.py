"""Inspect authenticated training ledger schema; no acoustic or timing inference."""
import collections
import hashlib
import json
from pathlib import Path

source = Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study/sources.json')
payload = source.read_bytes()
assert hashlib.sha256(payload).hexdigest() == '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
data = json.loads(payload)
engines, fields = collections.Counter(), collections.Counter()
for item in data:
    for alias in item['aliases']:
        if alias['label'] == 1:
            engines[alias['engine']] += 1
            fields.update(alias.keys())
report = {
    'scope': 'Schema census of authenticated historical training source ledger only.',
    'source': str(source), 'source_sha256': hashlib.sha256(payload).hexdigest(),
    'positive_aliases': sum(engines.values()), 'positive_aliases_by_engine': dict(engines),
    'positive_alias_field_counts': dict(fields),
    'target_interval_authenticated_by_this_ledger': False,
    'interpretation': 'PCM length/nonzero span/container geometry cannot authenticate a target interval within a long command. This ledger alone is insufficient for target-support endpoint substitution.',
    'next_evidence': 'Find hash-linked original generation timing or produce provenance-hashed automatic alignment on training-only PCM with declared synthetic-reference status; preserve every original source and explicit unsupported fallback. Enumerate endpoint feasibility before any gradient.',
    'limits': 'Does not establish absence of timing in other artifacts, phoneme errors, acoustic support, or impossibility of research. No human recordings/listening required; no training, inference, development/final data or promotion.',
}
Path('research/wakeword_p4/results/4090-positive-support-metadata-census-v1.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'positive_aliases': report['positive_aliases'], 'engines': dict(engines)}))
