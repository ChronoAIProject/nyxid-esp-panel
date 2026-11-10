"""Replay one existing training-negative phrase; never report held-out quality."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate import Detector, event_indices, smooth


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'features', 'calibration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--text', required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    calibration = json.loads(a.calibration.read_text())
    if (not calibration.get('validation_only') or calibration['threshold'] is None
            or calibration['model_sha256'] != sha(a.model)):
        raise ValueError('Need same-model fixed validation calibration')
    index = a.features/'features.json'
    # Variant zero is the original rendering, not a sample of scored augmentations.
    rows = [r for r in json.loads(index.read_text())
            if r['split'] == 'training' and r['label'] == 0
            and r['variant'] == 0 and r['text'] == a.text]
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Need unique original training-negative sources')
    raw = np.memmap(a.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    detector = Detector(a.model)
    records = []
    for row in rows:
        offset, frames = row['offset'], row['frames']
        if offset < 0 or frames < 150 or offset+frames > len(raw):
            raise ValueError('Invalid feature interval')
        data = raw[offset:offset+frames]
        detector.reset()
        scores = smooth(detector.predict(data), calibration['sliding_window'])
        detector.reset()
        np.testing.assert_array_equal(scores, smooth(detector.predict(data), calibration['sliding_window']))
        steps = event_indices(scores, calibration['threshold'])
        records.append(dict(path=row['path'], engine=row['engine'], source_sha256=row['sha256'],
                            peak=float(np.max(scores)), event_steps=[int(i) for i in steps],
                            scores_sha256=hashlib.sha256(scores.tobytes()).hexdigest()))
    groups = defaultdict(list)
    for row in records:
        groups[row['engine']].append(row)
    result = dict(role='Training-only original phrase replay; not held-out accuracy or acceptance',
                  model_sha256=sha(a.model), calibration_sha256=sha(a.calibration),
                  feature_index_sha256=sha(index), feature_binary_sha256=sha(a.features/'features.bin'),
                  source_sha256=sha(Path(__file__)), detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  text=a.text, threshold=calibration['threshold'], sliding_window=calibration['sliding_window'],
                  summaries={e:dict(clips=len(rs), triggered_clips=sum(bool(r['event_steps']) for r in rs),
                                    peak_max=max(r['peak'] for r in rs),
                                    peak_median=float(np.median([r['peak'] for r in rs]))) for e,rs in groups.items()},
                  records=records, reset_repeatability_passed=True,
                  independent_final_audio_accessed=False,
                  limits='Existing training variant-zero features, same frozen validation threshold. Correlated TTS sources; expected text is not independent phonetic truth. This is resubstitution, not generalization evidence. Selection by phrase may follow development errors; no held-out voice/audio is added to training and no labels or thresholds changed.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summaries']), flush=True)


if __name__ == '__main__':
    main()
