"""Attribute existing validation event results; never run inference or choose thresholds."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from evaluate import event_indices

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
report = json.loads((a.evaluation/'report.json').read_text())
traces = json.loads((a.evaluation/'clip-details.json').read_text())
if not report.get('validation_only') or traces['testing'] is not None:
    raise ValueError('This diagnostic accepts validation-only traces')
threshold = report['threshold']
groups = defaultdict(lambda: dict(variants=0, detected=0, premature=0))
positive_hits, negative_hits, positive_n, negative_n = 0, 0, 0, 0
for row in traces['validation']:
    events = event_indices(row['scores'], threshold)
    premature = 0
    if row['label']:
        end = row['keyword_end']
        hit = any(end-.1 <= row['times'][i] <= end+.8 for i in events)
        premature = sum(row['times'][i] < end-.1 for i in events)
        positive_hits += hit; positive_n += 1
    else:
        hit = bool(events); negative_hits += hit; negative_n += 1
    for kind, value in [('engine', row['engine']), ('variant', str(row['variant'])),
                        ('engine_variant', f"{row['engine']}/{row['variant']}"),
                        ('engine_phrase', f"{row['engine']}/{row['text']}")]:
        g = groups[(row['label'], kind, value)]
        g['variants'] += 1; g['detected'] += hit; g['premature'] += premature
validation = report['validation']
assert positive_hits == validation['detected'] and positive_n == validation['positive_variants']
assert abs(negative_hits/negative_n-validation['hard_negative_false_rate']) < 1e-12
summaries = [dict(label=label, grouping=kind, value=value, **g,
                  event_fraction=g['detected']/g['variants']) for (label, kind, value), g in groups.items()]
result = dict(role='Existing validation event attribution, no new threshold or inference',
              model_sha256=report['model_sha256'], threshold=threshold,
              report_sha256=hashlib.sha256((a.evaluation/'report.json').read_bytes()).hexdigest(),
              trace_sha256=hashlib.sha256((a.evaluation/'clip-details.json').read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              reproduced_positive_hits=positive_hits, reproduced_negative_hits=negative_hits,
              groups=summaries, final_or_regression_audio_scored=False,
              limits='Correlated synthetic variants; event fractions are recall for positives and false detections for negatives. No causal effect or final acceptance is established.')
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([x for x in summaries if x['label'] and x['grouping']=='engine_variant'], indent=2))
