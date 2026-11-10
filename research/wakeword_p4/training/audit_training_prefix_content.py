"""Describe source content retained by existing training-prefix negative crops.

Waveform energy is not word/phone completeness. No labels or training inputs
are changed, and no detector or final audio is loaded.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh report required')
    index = a.features/'features.json'
    groups = defaultdict(list)
    for row in json.loads(index.read_text()):
        if (row['split']=='training' and row['label']==1 and row['variant']==0
                and row['text'] in ('Hey Nyx.','Hey Nicks.','Hey Nix!','Hey Nicks?')):
            groups[row['engine']].append(row)
    if set(groups) != {'piper','kokoro','qwen-cuda'}:
        raise ValueError('Need all three existing training synthesis sources')
    results = []
    for engine,rows in sorted(groups.items()):
        rows = sorted(rows,key=lambda r:r['path'])
        if len({r['path'] for r in rows}) != len(rows):
            raise ValueError('Repeated original source path')
        selection = np.random.default_rng(10726).choice(len(rows),size=min(64,len(rows)),replace=False)
        for i in selection:
            row = rows[int(i)]
            path = a.features/'dataset'/row['path']
            if sha(path) != row['sha256'] or row['lead_frame'] != 200:
                raise ValueError('Changed source or unexpected unaugmented lead')
            pcm,rate = sf.read(path,dtype='int16')
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']:
                raise ValueError('Changed source PCM format/length')
            power = pcm.astype(np.float64)**2
            total = power.sum()
            if total <= 0:
                raise ValueError('Silent source')
            nonzero = np.flatnonzero(pcm)
            prefixes = []
            for gap in (15,25,35):
                end = row['keyword_frame']-gap
                # Feature[k] uses samples[k*160:(k+3)*160]; crop end is exclusive.
                start = (end-150)*160-row['lead_frame']*160
                stop = (end+2)*160-row['lead_frame']*160
                lo,hi = max(0,start),min(len(pcm),max(0,stop))
                prefixes.append(dict(gap_frames=gap,source_start_sample=start,source_end_sample=stop,
                                     retained_full_clip_energy_fraction=float(power[lo:hi].sum()/total),
                                     retains_every_nonzero_source_sample=bool(start<=nonzero[0] and stop>nonzero[-1])))
            results.append(dict(path=row['path'],engine=engine,source_sha256=row['sha256'],
                                keyword_end=row['keyword_end'],source_samples=len(pcm),prefixes=prefixes))
    summary = {}
    for engine in sorted(groups):
        summary[engine] = {}
        for gap in (15,25,35):
            prefixes = [p for r in results if r['engine']==engine for p in r['prefixes'] if p['gap_frames']==gap]
            summary[engine][str(gap)] = dict(clips=len(prefixes),
                retains_every_nonzero_source_sample=sum(p['retains_every_nonzero_source_sample'] for p in prefixes),
                energy_fraction_at_least_999=sum(p['retained_full_clip_energy_fraction']>=.999 for p in prefixes),
                median_retained_energy=float(np.median([p['retained_full_clip_energy_fraction'] for p in prefixes])))
    report = dict(role='Training-only prefix source-content probe, not phonetic correctness',
                  source_sha256=sha(Path(__file__)),feature_index_sha256=sha(index),
                  renderer_source_sha256=sha(Path(__file__).with_name('stream_data.py')),
                  seed=10726,clips_per_engine_max=64,variant=0,context_frames=150,gaps_frames=[15,25,35],
                  summary=summary,records=results,labels_changed=False,models_scored=False,
                  independent_final_accessed=False,
                  limits='Energy retention is not word/phone completeness; final fricatives can have little energy. Zero exact-content-retention cases do not validate negative labels or proxy endpoints. Original standalone source PCM only; rendered gain/clipping and augmentation not audited here. No classifier inference, relabeling, generalization or latency acceptance.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
