"""Attribute early events in already-scored development traces without relabeling."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate import event_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--split', choices=('validation', 'testing'), default='validation',
                   help='testing refers only to the existing development regression, never independent final')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report path')
    report = json.loads((a.evaluation/'report.json').read_text())
    rows = json.loads((a.evaluation/'clip-details.json').read_text())[a.split]
    expected = report['validation' if a.split == 'validation' else 'testing_regression']
    groups = defaultdict(lambda: dict(positive_variants=0, detected=0, premature_events=0,
                                     missed_with_valid_peak=0))
    offsets, details = [], []
    for row in rows:
        if not row['label']:
            continue
        indices = event_indices(row['scores'], report['threshold'])
        end = row['keyword_end']
        hits = [i for i in indices if end-.1 <= row['times'][i] <= end+.8]
        early = [i for i in indices if row['times'][i] < end-.1]
        g = groups[row['engine']]
        g['positive_variants'] += 1
        g['detected'] += bool(hits)
        g['premature_events'] += len(early)
        g['missed_with_valid_peak'] += not hits and row['max_score'] >= report['threshold']
        times = [(row['times'][i]-end)*1000 for i in early]
        offsets.extend(times)
        if early:
            details.append(dict(source=row['source'], variant=row['variant'], engine=row['engine'],
                                text=row['text'], keyword_end=end, event_offsets_ms=times,
                                later_valid_event=bool(hits), valid_window_peak=row['max_score']))
    for key in ('positive_variants', 'detected', 'premature_events'):
        assert sum(g[key] for g in groups.values()) == expected[
            'premature_positive_events' if key == 'premature_events' else key]
    result = dict(role='Existing development trace diagnostic; no inference or threshold changes',
                  split=a.split, model_sha256=report['model_sha256'], threshold=report['threshold'],
                  trace_sha256=hashlib.sha256((a.evaluation/'clip-details.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  by_engine=dict(groups), clips_with_early_events=details,
                  early_offset_ms_quantiles={str(q): float(np.quantile(offsets, q))
                                             for q in (0, .25, .5, .75, 1)} if offsets else {},
                  independent_final_accessed=False,
                  limits='Early events may reflect pronunciation, acoustic confusion or annotation error; timing alone does not establish cause or justify changing labels.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'clips_with_early_events'}, indent=2))


if __name__ == '__main__':
    main()
