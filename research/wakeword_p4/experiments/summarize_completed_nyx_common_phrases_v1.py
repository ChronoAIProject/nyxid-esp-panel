"""Aggregate saved events at the already-public .5 diagnostic point only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

R = Path('research/wakeword_p4/results')
BASE = Path('.local/wakeword/engine-adversarial-completed-nyx-development-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit_path = R / '4090-engine-adversarial-completed-nyx-scoring-independent-audit-v1.json'
    audit = json.loads(audit_path.read_text())
    collection_path = BASE / 'result.json'
    collection = json.loads(collection_path.read_text())
    assert audit['all_passed'] and collection['all_passed']
    evidence = {str(p): sha(p) for p in [audit_path, collection_path]}
    results = {}
    for model in ['hey_nyx-control', 'hey_nyx-adversarial']:
        path = BASE / 'diagnostic' / (model + '.jsonl')
        assert sha(path) == next(m['records_sha256'] for m in audit['models'] if m['model'] == model)
        report_path = BASE / model / 'report.json'
        assert sha(report_path) == next(m['report_sha256'] for m in collection['models'] if m['model'] == model)
        evidence.update({str(p): sha(p) for p in [path, report_path]})
        metadata = {(r['phase'], r['index']): r['record'] for r in json.loads(report_path.read_text())['records']}
        point = None
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record['summary']['threshold'] == .5:
                assert point is None
                point = record
        assert point is not None and len(point['records']) == 372
        groups = {}
        for field in ['literal', 'text']:
            totals = defaultdict(lambda: dict(clips=0, activated_clips=0, events=0, events_before_source=0))
            for item in point['records']:
                if item['phase'] != 'synthetic' or item['target_id'] in ['hey_nyx', 'hey_willow']:
                    continue
                row = metadata[(item['phase'], item['index'])]
                assert row['source_id'] == item['source_id'] and row['target_id'] == item['target_id']
                group = totals[row[field]]
                group['clips'] += 1
                group['activated_clips'] += bool(item['events'])
                group['events'] += len(item['events'])
                group['events_before_source'] += sum(e['sample'] < row['lead_samples'] for e in item['events'])
            assert sum(g['clips'] for g in totals.values()) == 288
            for key in ['activated_clips', 'events']:
                assert sum(g[key] for g in totals.values()) == point['summary']['negative']['common'][key]
            groups[field] = [dict(phrase=text, **values) for text, values in sorted(totals.items())]
        results[model] = dict(original_summary=point['summary'], grouped_common=groups)
    report = dict(schema='completed-nyx-common-phrase-saved-event-summary-v1',
                  script_sha256=sha(__file__), evidence=evidence, diagnostic_threshold=.5,
                  models=results, threshold_selected=False, detector_invocations=0,
                  new_audio_opened=False, limits='Descriptive grouping of original consumed diagnostic events at the .5 point already cited by Mac. No event recalculation, threshold selection, new inference or holdout use. Text association, including reported pre-source events, cannot isolate acoustic prefix versus suffix causes or certify phonetic truth. Failed intervention remains failed; no new candidate qualification.')
    with (R / '4090-completed-nyx-common-phrase-saved-event-summary-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({m: d['grouped_common']['literal'] for m, d in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
