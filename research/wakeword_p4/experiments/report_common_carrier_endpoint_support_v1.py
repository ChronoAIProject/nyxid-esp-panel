"""Describe frozen crop support from saved metadata; no sampler/model/RNG."""
import gzip
import hashlib
import json
from pathlib import Path

BASE = Path('.local/wakeword')
RESULTS = Path('research/wakeword_p4/results')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = BASE / 'hey-common-carrier-features-v1/features.json'
    new = {(r['anchor_id'], r['carrier'], r['variant']): r
           for r in json.loads(metadata.read_text())}
    inputs = {str(metadata): sha(metadata)}
    targets = []
    for target in ('hey_willow', 'hey_nyx'):
        ledger = BASE / 'hey-common-carrier-replacement-decisions-v1' / (target + '.jsonl.gz')
        inputs[str(ledger)] = sha(ledger)
        cells = {}
        selected = 0
        with gzip.open(ledger, 'rt') as stream:
            for batch in map(json.loads, stream):
                for choice in batch['choices']:
                    if not choice['replace']:
                        continue
                    selected += 1
                    key = (choice['anchor_id'], choice['carrier'], choice['variant'])
                    row = new[key]
                    old_lo = max(225, choice['old_lead'] - 25)
                    new_lo = max(225, row['lead_frame'] - 25)
                    old_n = choice['old_frames'] - old_lo + 1
                    new_n = row['frames'] - new_lo + 1
                    if key not in cells:
                        support = [new_lo + ((2 * i + 1) * new_n) // (2 * old_n)
                                   for i in range(old_n)]
                        assert support == sorted(support)
                        assert new_lo <= min(support) <= max(support) <= row['frames']
                        cells[key] = dict(anchor_id=key[0], carrier=key[1], variant=key[2],
                                          old_domain=[old_lo, choice['old_frames']],
                                          new_domain=[new_lo, row['frames']],
                                          reachable_count=len(set(support)), new_domain_count=new_n,
                                          reachable_min=min(support), reachable_max=max(support),
                                          reachable_sha256=hashlib.sha256(json.dumps(support).encode()).hexdigest(),
                                          selected_draws=0, observed=set())
                    cell = cells[key]
                    assert cell['old_domain'] == [old_lo, choice['old_frames']]
                    mapped = new_lo + choice['crop_midpoint_numerator'] * new_n // choice['crop_midpoint_denominator']
                    assert choice['crop_midpoint_numerator'] == 2 * (choice['old_end'] - old_lo) + 1
                    assert choice['crop_midpoint_denominator'] == 2 * old_n
                    cell['selected_draws'] += 1
                    cell['observed'].add(mapped)
        rows = []
        for key in sorted(cells):
            cell = cells[key]
            observed = cell.pop('observed')
            cell.update(observed_unique=len(observed), observed_min=min(observed), observed_max=max(observed))
            rows.append(cell)
        assert selected == 37114
        targets.append(dict(target=target, selected=selected, cells=rows,
                            cells_with_unreachable_new_frames=sum(r['reachable_count'] < r['new_domain_count'] for r in rows)))
    report = dict(inputs=inputs, source_sha256=sha(Path(__file__)), targets=targets,
                  limits='Descriptive support of unchanged integer midpoint mapping, selected cells only. Reachability is all legal old endpoints; observed counts are finite saved selected draws. No sampler, RNG, feature modification, model calls or recognition gate.')
    destination = RESULTS / '4090-hey-common-carrier-endpoint-support-v1.json'
    with destination.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in t.items() if k != 'cells'} for t in targets]))


if __name__ == '__main__':
    main()
