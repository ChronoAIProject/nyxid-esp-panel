"""Paired fixed-profile event comparison against saved C161 Melo traces."""
import hashlib
import json
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps

R = Path('research/wakeword_p4/results')
OUT = Path('.local/wakeword/bc01-melo-v1')
BASE = Path('.local/wakeword/melo-development-int8-v2/c161')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['all_passed'] and result['counts'] == {'bc01': 474}
    paths = [BASE / 'records.jsonl', OUT / 'bc01/records.jsonl']
    baseline = json.loads((BASE.parent / 'result.json').read_text())
    assert baseline['all_passed'] and sha(paths[0]) == baseline['record_files']['c161']
    assert sha(paths[1]) == result['record_files']['bc01']
    old, new = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    assert len(old) == len(new) == 474
    records = []
    groups = {}
    background = dict(hours=0., c161_events=0, bc01_events=0)
    for a, b in zip(old, new, strict=True):
        assert a['index'] == b['index'] and a['record'] == b['record']
        row = a['record']
        events = []
        for rec in [a, b]:
            path = Path(rec['trace_path'])
            assert sha(path) == rec['trace_sha256']
            with np.load(path) as trace:
                codes = trace['uint8']
                assert codes.dtype == np.uint8 and rec['prime_samples'] == 81600
                blocks = trace['block_return_sample'] - 81600
                steps = event_steps(codes, .91796875)
                events.append([dict(step=s, relative_sample=int(blocks[s-1])) for s in steps])
        real = [[e for e in model if e['relative_sample'] > 0] for model in events]
        if row['phase'] == 'background':
            background['hours'] += row['samples']/16000/3600
            background['c161_events'] += len(real[0]); background['bc01_events'] += len(real[1])
        else:
            group = row['voice_id'].split(':')[-1] + '/' + row['phrase_id'] + '/' + row['carrier']
            item = groups.setdefault(group, dict(clips=0, c161_hits=0, bc01_hits=0, gained=0, lost=0))
            before, after = map(bool, real)
            item['clips'] += 1; item['c161_hits'] += before; item['bc01_hits'] += after
            item['gained'] += after and not before; item['lost'] += before and not after
        records.append(dict(index=a['index'], source=row, c161_events=events[0], bc01_events=events[1]))
    report = dict(schema='bc01-melo-fixed-comparison-v1', inputs={str(p): sha(p) for p in paths},
        threshold=.91796875, smoothing_steps=5, groups=groups, background=background, records=records,
        policy='Consumed Melo fixed-profile whole-container events, no threshold search or new background hours. EN-BR is British. Synthetic prefix reference not phonetic truth. No automatic promotion or human/P4 claim.')
    (R / '4090-bc01-melo-fixed-comparison-v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(groups=groups, background=background), indent=2))


if __name__ == '__main__':
    main()
