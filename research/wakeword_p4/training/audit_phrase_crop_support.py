"""Audit complete original PCM support in training-only variant0 feature crops.

Coverage is an acoustic support property, not pronunciation or phonetic timing.
No classifier or validation/final audio is accessed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh report required')
    index = a.features / 'features.json'
    rows = [r for r in json.loads(index.read_text())
            if r['split'] == 'training' and r['variant'] == 0]
    seen, groups, records = set(), defaultdict(lambda: [0, 0]), []
    for r in rows:
        key = (r['engine'], r['path'])
        if key in seen:
            raise ValueError('Repeated original training source')
        seen.add(key)
        path = a.features / 'dataset' / r['path']
        if sha(path) != r['sha256']:
            raise ValueError('Source hash changed')
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != r['samples']:
            raise ValueError('Source PCM metadata changed')
        if r['lead_frame'] != 200:
            raise ValueError('Only exact variant0 lead is supported')
        # Feature k covers [160*k,160*(k+3)). For exclusive crop end e,
        # 150 frames cover [160*(e-150),160*(e+2)).
        lead = 32000
        end = (lead + info.frames + 159) // 160 - 2
        start_sample, stop_sample = (end - 150)*160, (end + 2)*160
        fits = (end >= 150 and end <= r['frames'] and
                start_sample <= lead and stop_sample >= lead + info.frames)
        group = (r['engine'], r['label'], r['text'])
        groups[group][0] += 1
        groups[group][1] += int(fits)
        records.append({'path': r['path'], 'engine': r['engine'], 'text': r['text'],
                        'label': r['label'], 'source_sha256': r['sha256'],
                        'samples': info.frames, 'entire_source_fits': fits,
                        'feature_crop_end': end if fits else None,
                        'feature_offset': r['offset']})
    result = {'role': 'Training-only complete-source crop support audit; not phone labels',
              'source_sha256': sha(Path(__file__)), 'feature_index_sha256': sha(index),
              'context_frames': 150, 'acoustic_support_samples': 24320,
              'sources': len(records), 'entire_source_fits': sum(r['entire_source_fits'] for r in records),
              'by_intended_phrase': [dict(engine=k[0], label=k[1], text=k[2], sources=v[0], entire_source_fits=v[1]) for k,v in sorted(groups.items())],
              'records': records, 'models_scored': False, 'final_data_accessed': False,
              'limits': 'Full original PCM support does not prove correct pronunciation or that keyword phones remain unchanged under frontend preprocessing. Long sources are retained as not fitting, not relabeled or silently trimmed. Variant0 only; no augmented or reverberation-tail guarantee.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','by_intended_phrase')}, indent=2))


if __name__ == '__main__':
    main()
