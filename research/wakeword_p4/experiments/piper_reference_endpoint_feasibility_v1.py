"""Conservative generation-reference geometry, not acoustic receptive-field proof."""
import hashlib
import json
from collections import Counter
from pathlib import Path

source = Path('research/wakeword_p4/results/4090-piper-feature-timing-mapping-v1.json')
d = json.loads(source.read_text())
summary = Counter(); variants = {}; rows = []
for r in d['rows']:
    # lead lies in [160*lead_frame, 160*lead_frame+159].
    # Retain legacy conservative sample-container ledger for an exclusive
    # feature slice [e-150:e]: start=(e-150)*160, end=(e+1)*160.
    # This is deliberately not a claim about physical receptive fields.
    lead_min = r['lead_frame']*160
    target_end_max = lead_min+159+r['generation_keyword_end']
    lo = max(150, (target_end_max+159)//160-1)
    hi = min(r['frames'], r['lead_frame']+150)
    dry_count = max(0, hi-lo+1)
    reverberant = r['variant'] in [2,3,4,5]
    status = 'rir_tail_not_yet_authenticated' if reverberant else ('target_reference_supported' if dry_count else 'no_legal_endpoint')
    summary[status] += 1
    v = variants.setdefault(str(r['variant']), Counter()); v[status] += 1
    rows.append({'feature_index':r['feature_index'],'source_sha256':r['sha256'],
                 'status':status,'dry_reference_legal_endpoint_count':dry_count,
                 'dry_reference_first':lo if dry_count else None,
                 'dry_reference_last':hi if dry_count else None,
                 'fallback_policy':'Retain original sampler draw unchanged if unsupported or unauthenticated.'})
assert len(rows)==60000
report={'scope':'All60000 Piper positive main-training rows; conservative generated-prefix reference interval [0,keyword_end].',
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'context_frames':150,'summary':dict(summary),'variants':{k:dict(v) for k,v in variants.items()},'rows':rows,
        'limits':'RIR rows explicitly unknown until tail provenance authenticated; dry-reference feasibility on those rows is not full support. No illegal row discarded or changed. Not full-source support, acoustic/phonetic truth, feature-binary authentication, or a sampler implementation. No gradients, inference, newdev/final data or promotion.'}
Path('research/wakeword_p4/results/4090-piper-reference-endpoint-feasibility-v1.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(report['summary']))
