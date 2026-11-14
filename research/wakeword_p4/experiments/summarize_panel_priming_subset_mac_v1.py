"""Summarize actual events, keeping all original thresholds and both paths."""
import argparse
from collections import Counter
import json
from pathlib import Path

from replay_panel_priming_mac_v1 import sha


def summarize(source, output):
    data = json.loads(source.read_text())
    curves = []
    for model in sorted({r['model'] for r in data['records']}):
        records = [r for r in data['records'] if r['model'] == model]
        for phase_index in range(2):
            points = []
            for index, point in enumerate(records[0]['phases'][phase_index]['points']):
                counts = Counter()
                for record in records:
                    row = record['source']
                    events = record['phases'][phase_index]['points'][index]
                    if events['threshold'] != point['threshold']:
                        raise ValueError('Inconsistent grids')
                    real = events['real_input_events']
                    before = [e for e in real if e['source_sample'] <= row['lead_samples']]
                    after = [e for e in real if e['source_sample'] > row['lead_samples']]
                    counts['pre_source_clips'] += bool(before)
                    counts['ignored_priming_events'] += len(events['ignored_priming_events'])
                    if row['target_id'] == 'hey_nyx':
                        counts['positive_clips'] += bool(after)
                        counts['positive_bare'] += bool(after) and not row['carrier']
                        counts['positive_command'] += bool(after) and bool(row['carrier'])
                    else:
                        counts['cross_clips'] += bool(real)
                        counts['cross_events'] += len(real)
                points.append(dict(threshold=point['threshold'], **counts))
            clear = [p for p in points if not p['pre_source_clips'] and not p['cross_clips']]
            curves.append(dict(model=model, phase=records[0]['phases'][phase_index]['name'],
                               points=points, zero_subset_negative_points=len(clear),
                               positive_range_at_zero_subset_negatives=(
                                   [min(p['positive_clips'] for p in clear),
                                    max(p['positive_clips'] for p in clear)] if clear else None),
                               bare_range_at_zero_subset_negatives=(
                                   [min(p['positive_bare'] for p in clear),
                                    max(p['positive_bare'] for p in clear)] if clear else None)))
    result = dict(schema='mac-existing-priming-subset-summary-v1',
                  source_sha256=sha(source), script_sha256=sha(__file__),
                  source_model_records=len(data['records']), clips_per_model=data['clips'],
                  positive_clips=36, positive_bare=9, positive_command=27, cross_clips=72,
                  background_hours=0, candidate_promoted=False,
                  limits='Consumed synthetic subset only. Hits are events after source starts, not validated word-end latency. No background FAR, independent acceptance, acoustic or P4 qualification.',
                  curves=curves)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in c.items() if k != 'points'} for c in curves], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
