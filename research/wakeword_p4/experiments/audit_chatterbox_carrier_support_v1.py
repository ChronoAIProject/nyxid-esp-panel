"""Metadata-only source support census; no model/acceptance access."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('research/wakeword_p4/results')
SOURCE = ROOT / '4090-chatterbox-training-admission-v1.json'
OUT = ROOT / '4090-chatterbox-carrier-support-audit-v1.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = json.loads(SOURCE.read_text())
    records = []
    counts = Counter()
    seen = set()
    for disposition in ('admitted', 'quarantined'):
        for row in source[disposition]:
            assert row['index'] not in seen
            seen.add(row['index'])
            req = row['request']
            assert req['role'] == 'training-only'
            wav = row['wav']
            path = Path(wav['path'])
            assert path.stat().st_size == wav['bytes'] and sha(path) == wav['sha256']
            carrier = ('bare' if req['text'] == 'Hey Nicks.' else 'command') if req['requested_label'] else 'daily-negative'
            counts[disposition, req['voice'], carrier] += 1
            records.append(dict(index=row['index'], disposition=disposition,
                                voice=req['voice'], carrier=carrier, wav=wav))
    assert seen == set(range(96))
    result = dict(schema='chatterbox-carrier-support-audit-v1',
        inputs={str(SOURCE): sha(SOURCE), str(Path(__file__)): sha(Path(__file__))},
        counts=[dict(disposition=d, voice=v, carrier=c, count=n)
                for (d,v,c),n in sorted(counts.items())], records=records,
        verified_waveforms=96,
        limits='Actual admitted Chatterbox expansion only, not census of parent training pools. Metadata carrier is not phonetic truth. No inference, scoring, fit, new synthesis or acceptance access. Quarantine unchanged; no causal attribution of development misses.')
    with OUT.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['counts'], indent=2))

if __name__ == '__main__':
    main()
