"""Replay a frozen training-source queue for numeric exercise, without audio I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wakeword_p4/training'))
from evaluate import Detector
from features import features
from detector_policy import smooth
from prepare_runtime_selftest import events


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'plan', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads((Path(__file__).parent/'plans/gpu-v30-runtime-v1.json').read_text())
    assert sha(args.model) == profile['model_sha256']
    plan = json.loads(args.plan.read_bytes())
    index = args.features/'features.json'
    assert sha(index) == plan['feature_index_sha256']
    records = json.loads(index.read_bytes())
    groups = {}
    for row in records:
        if row['split'] == 'training' and row['variant'] == 0 and row['label'] == 1:
            groups.setdefault(row['engine'], []).append(row)
    expected = [(engine, row) for engine, rows in sorted(groups.items())
                for row in sorted(rows, key=lambda r: r['path'])[:2]]
    assert len(expected) == len(plan['sources'])
    for item, (engine, row) in zip(plan['sources'], expected, strict=True):
        assert Path(item['path']).resolve() == (args.features/'dataset'/row['path']).resolve()
        assert item['engine'] == engine and item['split'] == 'training'
        assert item['sha256'] == row['sha256'] == sha(Path(item['path']))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.plan, args.output/'plan.json')
    shutil.copyfile(__file__, args.output/'source.py')
    rows, arrays = [], {}
    prime = (profile['input_frames']+200)*160
    prime += (-prime) % 480
    for index, item in enumerate(plan['sources']):
        pcm, rate = sf.read(item['path'], dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) == item['samples']
        pcm = np.pad(pcm, (prime, (-len(pcm)) % 480)).astype('<i2')
        detector = Detector(args.model)
        scores = detector.predict(features(pcm))
        detector.reset()
        np.testing.assert_array_equal(scores, detector.predict(features(pcm)))
        assert len(scores) == len(pcm)//480-1
        arrays[f'scores_{index}'] = scores
        rows.append({**item, 'pcm_samples_with_prime_and_tail': len(pcm),
            'output_count': len(scores), 'unique_output_bytes': len(np.unique(np.rint(scores*256))),
            'minimum_score': float(scores.min()), 'maximum_score': float(scores.max()),
            'reset_reproducible': True,
            'host_event_steps': events(smooth(scores, profile['smoothing_steps']),
                                       profile['threshold'], profile['smoothing_steps'])})
    result = {'scope': plan['scope'], 'plan_sha256': sha(args.plan), 'sources': rows,
              'model_sha256': profile['model_sha256'], 'accepted': False}
    np.savez_compressed(args.output/'scores.npz', **arrays)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'sources': len(rows), 'reset_reproducible': True,
                      'accepted': False, 'scope': plan['scope']}))


if __name__ == '__main__':
    main()
