"""Independently reconstruct ordinary-positive supervision from retained draws."""
from collections import Counter
import hashlib
import json

import numpy as np


def audit(directory, features, steps):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    training = json.loads((directory/'training.json').read_bytes())
    positive = training['phone_distillation']['ordinary_positive']
    exposure = positive['exposure']
    path = directory/exposure['trace_file']
    require(path.name == 'positive-exposure.jsonl' and path.parent == directory, 'Wrong positive trace path')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == exposure['trace_sha256'], 'Positive trace digest differs')
    records = json.loads((features/'features.json').read_bytes())
    raw = np.memmap(features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    counts, sources, renders = Counter(), set(), set()
    number = 0
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            item = json.loads(line)
            require(item['step'] == number and len(item['draws']) == 24, 'Positive draw count/order differs')
            for index, end, selected, crop_sha in item['draws']:
                require(type(index) is int and 0 <= index < len(records), 'Invalid positive source index')
                row = records[index]
                require(row['split'] == 'training' and row['label'] == 1, 'Nontraining/nonpositive draw')
                require(type(end) is int and 335 <= end <= row['frames'], 'Invalid positive crop end')
                complete = end-335 <= row['lead_frame'] and end >= row['keyword_frame']+3
                require(type(selected) is int and selected in (0, 1) and bool(selected) == complete,
                        'Positive crop completeness marker differs')
                crop = raw[row['offset']+end-335:row['offset']+end]
                require(hashlib.sha256(crop.tobytes()).hexdigest() == crop_sha, 'Actual positive crop bytes differ')
                counts[(row['engine'], row['text'], complete)] += 1
                if complete:
                    sources.add(row['sha256'])
                    renders.add((row['sha256'], row['variant']))
    expected = {'steps': number, 'examples': sum(counts.values()),
        'eligible_examples': sum(n for (_, _, selected), n in counts.items() if selected),
        'eligible_distinct_sources': len(sources), 'eligible_distinct_renders': len(renders),
        'counts': [{'engine': e, 'text': text, 'eligible': selected, 'examples': n}
                   for (e, text, selected), n in sorted(counts.items())]}
    require(all(exposure[k] == v for k, v in expected.items()), 'Positive exposure totals differ')
    require(number == steps and positive['steps'] == steps and positive['target'] == [1, 2]
            and positive['examples'] == expected['eligible_examples'], 'Applied positive CTC counts differ')
    return {**expected, 'trace_sha256': digest, 'all_draw_crops_and_markers_reconstructed': True}
