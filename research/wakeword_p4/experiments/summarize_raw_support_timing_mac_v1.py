"""Describe every original timing group over all points and all FAR points."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def span(values):
    return [min(values), max(values)] if values else [None, None]


def summarize(source):
    data = source.read_bytes()
    raw = gzip.decompress(data) if source.suffix == '.gz' else data
    report = json.loads(raw)
    assert report['schema'] == 'mac-original-paired-raw-support-timing-v1'
    assert report['checked_points'] == len(report['points']) == 552
    models = defaultdict(list)
    for point in report['points']:
        models[point['model']].append(point)
    assert len(models) == 4 and all(len(points) == 138 for points in models.values())
    regions = []
    for model, points in models.items():
        for region, selected in [('all138', points), ('all_far_qualified', [p for p in points if p['far_requirement_satisfied']])]:
            groups = defaultdict(list)
            for point in selected:
                for group in point['groups']:
                    groups[(group['role'], group['literal'], group['carrier'])].append(group)
            rows = []
            for (role, literal, carrier), gs in sorted(groups.items()):
                rows.append(dict(role=role, literal=literal, carrier=carrier,
                    clips_per_point=span([g['clips'] for g in gs]),
                    events_per_point=span([g['events'] for g in gs]),
                    activated_per_point=span([g['activated_clips'] for g in gs]),
                    events_by_support_band_per_point={band: span([g['events_by_support_band'][band] for g in gs]) for band in gs[0]['events_by_support_band']},
                    first_event_signed_prefix_ms_range=span([g['first_event_minus_generated_prefix_ms'][key] for g in gs for key in ('minimum', 'maximum') if g['first_event_minus_generated_prefix_ms'][key] is not None])))
            regions.append(dict(model=model, region=region, points=len(selected), groups=rows))
    return dict(schema='mac-original-paired-raw-support-regions-v1',
        input_report_sha256=hashlib.sha256(raw).hexdigest(),
        input_reader_sha256=report['source_sha256'], all_regions=regions,
        threshold_selected=False, model_calls=0,
        limits='Descriptive ranges from every retained point/group, not independent trials or inferred phonetic boundaries. Empty FAR regions stay empty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.timing)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(regions=len(result['all_regions']), input_report_sha256=result['input_report_sha256'])))


if __name__ == '__main__':
    main()
