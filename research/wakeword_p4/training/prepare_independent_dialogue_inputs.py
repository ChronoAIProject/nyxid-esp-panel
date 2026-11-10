"""Freeze same-PCM independent dialogue WAVs and continuous frontend caches.

No model is loaded. Each meeting keeps its original pauses and overlaps;
only 0..159 final zero samples align the shared waveform to the frontend.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend
import soundfile as sf

from features import features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(job):
    source, output, row = job
    source, output = Path(source), Path(output)
    path = (source/row['path']).resolve()
    if (not path.is_relative_to(source.resolve()) or sha(path) != row['sha256']
            or row['split'] != 'testing' or row['role'] != 'independent-dialogue-final-source'):
        raise ValueError('Changed or non-final source')
    info = sf.info(path)
    pcm, rate = sf.read(path, dtype='int16')
    if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples'] or info.subtype != 'PCM_16'
            or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']):
        raise ValueError('Changed original PCM')
    padding = (-len(pcm)) % 160
    signal = np.pad(pcm, (0, padding))
    target = output/row['path']
    sf.write(target, signal, 16000, subtype='PCM_16')
    decoded, rate = sf.read(target, dtype='int16')
    np.testing.assert_array_equal(decoded, signal)
    # Share one frontend for the whole meeting, including at all 10s boundaries.
    frontend = MicroFrontend()
    chunks = [features(signal[at:at+160000], frontend) for at in range(0, len(signal), 160000)]
    data = np.concatenate(chunks)
    if len(chunks) < 2 or len(data) != len(signal)//160-2:
        raise ValueError('Invalid continuous frontend cadence')
    np.testing.assert_array_equal(np.concatenate(chunks[:2]), features(signal[:320000]))
    feature_path = Path('features')/(row['session']+'.npy')
    np.save(output/feature_path, data)
    return dict(row, role='independent-dialogue-final',
                original_wav_sha256=row['sha256'], original_pcm_sha256=row['pcm_sha256'],
                original_samples=len(pcm), source_seconds=len(pcm)/16000,
                frontend_final_zero_pad_samples=padding,
                samples=len(signal), seconds=len(signal)/16000,
                sha256=sha(target), pcm_sha256=hashlib.sha256(signal.astype('<i2').tobytes()).hexdigest(),
                feature_path=str(feature_path), feature_sha256=sha(output/feature_path),
                frames=len(data), first_two_chunk_continuity_checked=True,
                exact_original_pcm_preserved=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-audit-sha256', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.workers <= 8:
        p.error('Fresh output and1..8 workers required')
    audit_path = a.source/'audit.json'
    audit = json.loads(audit_path.read_text())
    if (sha(audit_path) != a.source_audit_sha256
            or sha(a.source/'manifest.jsonl') != audit['manifest_sha256']
            or audit['classifiers_scored'] or audit['waveform_edited']
            or not audit['training_and_development_participants_disjoint']):
        raise ValueError('Need pinned untouched independent source audit')
    rows = [json.loads(line) for line in (a.source/'manifest.jsonl').read_text().splitlines()]
    if (rows != audit['records'] or len(rows) != 16
            or sorted(r['session'] for r in rows) != audit['sessions']
            or len({r['path'] for r in rows}) != len(rows)
            or len({x for r in rows for x in r['participants']}) != audit['unique_participants']):
        raise ValueError('Changed full selected population')
    a.output.mkdir(parents=True)
    (a.output/'features').mkdir()
    records = []
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = [(str(a.source),str(a.output),r) for r in rows]
        for row in pool.map(worker, jobs):
            records.append(row)
            print(json.dumps(dict(completed=len(records),planned=len(rows),session=row['session'])),flush=True)
    report = dict(role='Independent dialogue same-PCM inputs; no classifier inference',
                  source_sha256=sha(Path(__file__)), source_audit_sha256=sha(audit_path),
                  source_manifest_sha256=audit['manifest_sha256'],
                  frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  source_hours=sum(r['source_seconds'] for r in records)/3600,
                  unique_participants=audit['unique_participants'], records=records,
                  continuous_recordings=True, inserted_intermediate_gaps=False,
                  original_pcm_preserved=True, classifiers_scored=False,
                  independent_final_audio_accessed_for_preparation=True,
                  policy='One whole meeting per detector reset. Identical PCM for both engines, including only final0..159 frontend alignment zeros. Count all events; exposure excludes zeros. Official final512-call padding recorded separately. No trimming, splicing or per-window reset.',
                  limits='Unscored source preparation only. Repeated meeting participants and public-transcript limitations retained. Requires separate committed model/threshold/engine/scorer preregistration before inference; not positive recall or latency evidence.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
