"""Whole-wave training cache, continuous81600 prime and fixed8000 tail."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from causal_sequence_supervision_v1 import masks

R = Path('research/wakeword_p4/results')
ADMISSION = R / '4090-chatterbox-training-admission-v1.json'
PLAN = R / '4090-chatterbox-training-cache-plan-v1.json'
OUT = Path('.local/wakeword/chatterbox-training-cache-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan():
    admission = json.loads(ADMISSION.read_text())
    rows = admission['admitted']
    assert len(rows) == 84 and sum(r['request']['requested_label'] for r in rows) == 36
    paths = [Path(__file__), ADMISSION, Path(__file__).with_name('causal_sequence_supervision_v1.py'),
             Path('research/wakeword_p4/training/features.py')]
    for row in rows:
        path = Path(row['wav']['path'])
        assert sha(path) == row['wav']['sha256']
        paths.append(path)
    record = dict(files={str(p): sha(p) for p in paths}, rows=rows, lead=81600, tail=8000,
        policy='Training-only whole waveforms, no crop/noise/pitch. Continuous81600 zero prime,8000 zero tail. Positives use independently supplied MFA prefix end; standard C161 masks: positive prefixend+30..270ms, negatives only before source start. Commands after prefix otherwise unsupervised. Negatives supervise full stream. Fixed source expansion, not independent voice evidence or acceptance.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show',
        'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    record = json.loads(PLAN.read_text())
    for path, expected in record['files'].items():
        assert sha(Path(path)) == expected
    import soundfile as sf
    sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    OUT.mkdir(exist_ok=False)
    rows = []
    offset = 0
    with (OUT / 'features.bin').open('xb') as stream:
        for source in record['rows']:
            pcm, rate = sf.read(source['wav']['path'], dtype='int16')
            assert rate == 16000 and pcm.ndim == 1
            combined = np.concatenate((np.zeros(81600, np.int16), pcm, np.zeros(8000, np.int16)))
            raw = features(combined)
            label = source['request']['requested_label']
            endpoint = 81600 + round(source['automatic_prefix_end_seconds'] * 16000) if label else None
            support, positive, negative = masks(len(raw), label, 81600 if label else None, endpoint)
            assert len(raw) <= 846 and len(positive) <= 282
            assert positive.sum() == 0 if not label else np.isclose(positive.sum(), 1)
            assert np.isclose(negative.sum(), 1) and not np.any((positive > 0) & (negative > 0))
            raw.astype('<u2').tofile(stream)
            rows.append(dict(source=source, offset=offset, frames=len(raw), label=label,
                voice=source['request']['voice'], lead=81600 if label else None, endpoint=endpoint,
                feature_sha256=hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest(),
                positive_calls=np.flatnonzero(positive).tolist(), negative_calls=int(np.count_nonzero(negative))))
            offset += len(raw)
    (OUT / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
    result = dict(plan_sha256=sha(PLAN), records=len(rows), frames=offset,
                  files={name: sha(OUT / name) for name in ['features.bin', 'records.json']},
                  bytes=(OUT / 'features.bin').stat().st_size, all_masks_checked=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    {'plan': plan, 'run': run}[sys.argv[1]]()
