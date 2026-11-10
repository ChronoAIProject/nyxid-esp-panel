"""Mine training-only hard windows from the frozen diagnostic classifier."""
from pathlib import Path
import argparse
import json
import hashlib
import time
import numpy as np

from train_head import load_rows
from evaluate_head import Head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--teacher-plan', type=Path, required=True)
    parser.add_argument('--backbone-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    P = args.data_root
    output = args.output
    assert not output.exists()
    rows = load_rows('training', P, args.backbone_sha256)
    rows = [r for r in rows if not r['label']]
    frozen = json.loads(args.teacher_plan.read_text())['selected']
    model = Path(frozen['checkpoint'])/'head.tflite'
    assert hashlib.sha256(model.read_bytes()).hexdigest() == frozen['model_sha256']
    detector = Head(model)
    selected, counts = [], {}
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        scores, times = detector.scores(row['array'], 1)
        ends = np.rint(times/.08).astype(np.int64)
        candidates = np.flatnonzero(scores >= .05)
        picked = []
        if row['kind'] == 'tts':
            for index in candidates[np.argsort(scores[candidates])[::-1]]:
                if all(abs(int(ends[index])-int(ends[previous])) >= 4 for previous in picked):
                    picked.append(int(index))
                if len(picked) == 2:
                    break
        else:
            # Keep at most one peak per second and 256 distinct bins per source.
            by_second = {}
            for index in candidates:
                second = int(times[index])
                previous = by_second.get(second)
                if previous is None or scores[index] > scores[previous]:
                    by_second[second] = int(index)
            picked = sorted(by_second.values(), key=lambda index: -scores[index])[:256]
        for index in picked:
            selected.append({'id': row['id'], 'end': int(ends[index]), 'score': float(scores[index]),
                             'kind': row['kind'], 'engine': row['engine'],
                             'feature_sha256': row['features_sha256']})
            group = row['kind']+'/'+row['engine']
            counts[group] = counts.get(group, 0)+1
        if number % 1000 == 0 or number == len(rows):
            print(json.dumps({'completed': number, 'total': len(rows), 'selected': len(selected),
                              'seconds': time.monotonic()-started}), flush=True)
    result = {'scope': 'Training-only cached hard negatives; no validation/test audio fitted',
              'teacher_head_sha256': frozen['model_sha256'],
              'backbone_sha256': args.backbone_sha256,
              'feature_indexes': {name: hashlib.sha256((P/name/'index.jsonl').read_bytes()).hexdigest()
                                  for name in ['head-features-tts-v1', 'head-features-negatives-v1']},
              'policy': {'raw_score_minimum': .05, 'tts_max_per_variant': 2,
                         'tts_separation_steps': 4, 'continuous_peaks_per_second': 1,
                         'continuous_max_per_source': 256, 'warm_end_minimum': 25},
              'counts': counts, 'records': selected, 'seconds': time.monotonic()-started}
    output.write_text(json.dumps(result)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
