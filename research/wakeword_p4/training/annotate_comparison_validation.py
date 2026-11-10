"""Freeze energy-proxy endpoints before paired development wake scoring."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use fresh directory')
    source = json.loads(a.manifest.read_text())
    if not source.get('paired_wakephrases') or source.get('independent_final_audio_accessed'):
        raise ValueError('Need development paired corpus')
    records = []
    for row in source['records']:
        if row['role'] != 'development-paired-validation' or row['split'] != 'validation':
            raise ValueError('Only existing development validation')
        path = (a.manifest.parent/row['path']).resolve()
        if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed/escaped audio')
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples'] or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']:
            raise ValueError('Changed PCM')
        signal = pcm.astype(np.float32)/32768
        count = len(signal)//160
        rms = np.sqrt(np.mean(signal[:count*160].reshape(-1, 160)**2, axis=1))
        active = np.flatnonzero(rms > max(1e-6, float(rms.max())*.005))
        if not len(active):
            raise ValueError('No active speech')
        records.append(dict(row, keyword_end=min(len(pcm), (int(active[-1])+1)*160+960),
                            keyword_end_method='Last10ms RMS frame above max(1e-6,peak*.005) plus60ms; PCM retained; not phonetic truth'))
    a.output.mkdir(parents=True)
    for row in records:
        dest = a.output/row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.manifest.parent/row['path'], dest)
    report = dict(source, records=records, parent_manifest_sha256=sha(a.manifest),
                  endpoint_annotation_source_sha256=sha(Path(__file__)),
                  endpoint_protocol='Energy proxy only; fixed detection window endpoint-100ms through endpoint+800ms. No trimming, score-informed alignment or phonetic-truth claim.',
                  wake_scores_computed=False)
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(records=len(records), manifest_sha256=sha(a.output/'manifest.json'),
                          parent_manifest_sha256=sha(a.manifest), wake_scores_computed=False)))


if __name__ == '__main__':
    main()
