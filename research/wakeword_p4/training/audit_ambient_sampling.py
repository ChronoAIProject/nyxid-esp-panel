"""Audit source exposure versus sampling probabilities on training metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ambient_sampling import probabilities, coverage

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--features', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = []
for name in ['human.json', 'ambient.json']:
    rows.extend(r for r in json.loads((a.features/name).read_text()) if r['split'] == 'training')
groups = coverage(rows)
weights = probabilities(rows)
indices = np.random.default_rng(726).choice(len(rows), size=100000, p=weights)
human = np.array(['speaker' in r for r in rows])
observed = float(human[indices].mean())
expected = float(weights[human].sum())
assert abs(observed-expected) < .005
report = dict(role='Training ambient exposure/sampler audit; no recognition evidence',
              metadata_sha256={n: hashlib.sha256((a.features/n).read_bytes()).hexdigest()
                               for n in ['human.json', 'ambient.json']},
              groups=groups, monte_carlo_draws=100000, observed_human_fraction=observed,
              source_sha256={n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                             for n in ['ambient_sampling.py', 'audit_ambient_sampling.py']},
              final_or_regression_audio_scored=False)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
