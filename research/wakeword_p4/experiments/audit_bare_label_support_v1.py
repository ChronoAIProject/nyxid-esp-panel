"""Census existing provisional labels; never change admission or read wake scores."""
import collections
import hashlib
import json
from pathlib import Path

SOURCE = Path('research/wakeword_p4/results/4090-chatterbox-bare-training-admission-v1.json')
OUT = Path('research/wakeword_p4/results/4090-bare-label-support-census-v1.json')


def main():
    data = json.loads(SOURCE.read_text())
    cells = collections.defaultdict(collections.Counter)
    transcripts = collections.defaultdict(collections.Counter)
    rows = []
    for status in ('admitted', 'quarantined'):
        for row in data[status]:
            wav = Path(row['wav']['path'])
            assert wav.stat().st_size == row['wav']['bytes']
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            assert digest == row['wav']['sha256'] == row['asr']['sha256']
            request = row['request']
            assert request['role'] == 'training-only'
            cells[(request['text'], request['voice'])][status] += 1
            transcripts[status][row['asr']['transcript']] += 1
            rows.append(dict(index=row['index'], status=status, text=request['text'],
                             voice=request['voice'], seed=request['seed'],
                             transcript=row['asr']['transcript'], wav_sha256=digest))
    assert len(rows) == len({r['index'] for r in rows}) == 48
    assert len(cells) == 12 and all(sum(c.values()) == 4 for c in cells.values())
    result = dict(schema='bare-label-support-census-v1',
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cells=[dict(text=text, voice=voice, admitted=c['admitted'],
                              quarantined=c['quarantined']) for (text, voice), c in sorted(cells.items())],
                  transcripts={s: dict(c) for s, c in transcripts.items()}, rows=sorted(rows, key=lambda r:r['index']),
                  interpretation='Fixed balanced generation requests become unbalanced after provisional ASR screening. Text agreement is not phone truth. No spelling winner, significance, causal pronunciation explanation, new admission, or wake-score filtering inferred. All48 original WAV identities verified; no generation/inference/training.')
    with OUT.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['cells']))


if __name__ == '__main__':
    main()
