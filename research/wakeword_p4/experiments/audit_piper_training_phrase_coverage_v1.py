"""Training metadata only: conditioning-ID/phrase coverage, not physical people."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')
assert sha(b/'sources.json')=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
assert sha(b/'records.json')=='757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
s=json.loads((b/'sources.json').read_text());r=json.loads((b/'records.json').read_text())
phrases=['hey nyx','hey nick','hey next','hey mix'];ids=np.random.default_rng(726).permutation(904)[:672].tolist();allowed=set(ids)
groups={}
for phrase in phrases:
 rows=[(i,x['aliases'][0]) for i,x in enumerate(s) if x['aliases'][0]['engine']=='piper' and x['aliases'][0]['phrase']==phrase]
 voices={int(v) for _,x in rows for v in x['speakers']};assert not voices-allowed
 groups[phrase]=dict(canonical_sources=len(rows),conditioning_ids=len(voices),missing_training_ids=sorted(allowed-voices),source_indices=[i for i,_ in rows],counts={str(v):sum(v in x['speakers'] for _,x in rows) for v in ids})
report=dict(scope='All inherited training-source Piper aliases; conditioning IDs, not acoustic/physical-person identity',source_sha256=sha(b/'sources.json'),records_sha256=sha(b/'records.json'),partition_seed=726,training_ids=ids,pilot_ids=ids[:32],groups=groups,script_sha256=sha(Path(__file__)),limits='Pair membership counts both conditioning IDs; not independent voice or acoustic realization proof. No development/final manifest, waveform, feature or score accessed; no wake model inference.')
Path('research/wakeword_p4/results/4090-piper-training-phrase-coverage-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({p:{k:v for k,v in g.items() if k in ['canonical_sources','conditioning_ids']} for p,g in groups.items()}))
