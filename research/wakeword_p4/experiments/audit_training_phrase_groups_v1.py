"""Inventory training-only same-voice phrase coverage without loading any scores."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path('.local/wakeword')
OUT = Path('research/wakeword_p4/results/4090-training-phrase-group-coverage-v1.json')
PHRASES = {'hey_nyx', 'hey_nick', 'hey_next', 'hey_mix'}


def main():
    reports = {}
    for pack in ['british-training-features-v1', 'factorized-training-features-v1']:
        path = BASE / pack / 'report.json'
        data = json.loads(path.read_text())
        rows = [r for r in data['records'] if r.get('pack_index') is not None]
        assert all(r['split'] == 'training' for r in rows)
        groups = defaultdict(Counter)
        for row in rows:
            speed = row.get('speed', row.get('length_multiplier'))
            condition = row.get('arm', row.get('variant'))
            groups[(str(row['voice']), str(speed), str(condition))][row['target']] += 1
        missing = []
        duplicates = []
        for key, counts in sorted(groups.items()):
            if PHRASES - counts.keys():
                missing.append(dict(voice=key[0], speed=key[1], condition=key[2],
                                    missing=sorted(PHRASES - counts.keys()), counts=dict(counts)))
            if any(n != 1 for n in counts.values()):
                duplicates.append(dict(key=key, counts=dict(counts)))
        reports[pack] = dict(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             included_records=len(rows), voices=len({r['voice'] for r in rows}),
                             phrase_counts=dict(Counter(r['target'] for r in rows)),
                             groups=len(groups), incomplete_groups=missing,
                             nonunit_groups=duplicates)
    result = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  expected_phrase_set=sorted(PHRASES), packs=reports,
                  candidate_inference=False, development_accessed=False,
                  limits='Metadata coverage only. Equal named phrase coverage does not establish acoustic '
                  'quality, effective optimization weight, phonetic truth, or cross-source generalization. '
                  'This inventory does not allocate new data or authorize recycling development into training.')
    with OUT.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: {a: len(b) if isinstance(b, list) else b for a, b in v.items()}
                      for k, v in reports.items()}))


if __name__ == '__main__':
    main()
