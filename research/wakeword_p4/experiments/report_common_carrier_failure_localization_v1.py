"""Localize failures over every retained threshold; never select a threshold."""
import hashlib
import json
from pathlib import Path

ROOT = Path('research/wakeword_p4/results')
source = ROOT / '4090-hey-common-carrier-raw-grouped-diagnostic-v1.json'
raw = source.read_bytes()
x = json.loads(raw)
rows = []
for model in x['models']:
    points = [p for p in model['points'] if p['far_requirement_satisfied']]
    result = {'model': model['model'], 'far_qualified_points': len(points), 'literal_ranges': [], 'persistent_common_sources': []}
    if points:
        for role in ('common', 'reciprocal', 'target'):
            names = sorted({g['value'] for p in points for g in p['groups'] if g['role'] == role and g['dimension'] == 'literal'})
            for name in names:
                groups = [g for p in points for g in p['groups'] if g['role'] == role and g['dimension'] == 'literal' and g['value'] == name]
                assert len(groups) == len(points)
                result['literal_ranges'].append({'role': role, 'literal': name, 'clips': groups[0]['clips'], 'activated_clips_range': [min(g['activated_clips'] for g in groups), max(g['activated_clips'] for g in groups)]})
        sets = [{g['value'] for g in p['groups'] if g['role'] == 'common' and g['dimension'] == 'source_id' and g['activated_clips'] > 0} for p in points]
        result['persistent_common_sources'] = sorted(set.intersection(*sets))
    rows.append(result)
report = {'schema': 'common-carrier-failure-localization-v1', 'source_sha256': hashlib.sha256(raw).hexdigest(), 'models': rows, 'model_invocations': 0, 'threshold_selected': False, 'limits': 'Ranges/intersections across ALL background-FAR-qualified points, not a selected point. Correlated variants, single consumed Semaine source; no causal claim or fresh acceptance. Source persistence means at least one variant triggers at each point, not necessarily the same variant.'}
out = ROOT / '4090-hey-common-carrier-failure-localization-v1.json'
with out.open('x') as f:
    f.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(rows, indent=2))
