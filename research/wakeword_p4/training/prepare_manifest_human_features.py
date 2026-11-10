"""Extract continuous MicroFrontend streams from an audited WAV manifest."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from pymicro_features import MicroFrontend
from features import features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(job):
    source, output, rows = job
    source, output = Path(source), Path(output)
    frontend, chunks, prefix, digest = MicroFrontend(), [], [], hashlib.sha256()
    source_samples, total_samples = 0, 0
    for row in sorted(rows, key=lambda r: r['path']):
        path = (source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed or escaped source WAV')
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']:
            raise ValueError('Changed PCM format/length')
        if hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']:
            raise ValueError('Changed decoded PCM')
        source_samples += len(pcm)
        # Explicit alignment padding is counted as inserted silence, never exposure.
        gap = 3200+(-len(pcm)-3200) % 160
        signal = np.concatenate((pcm, np.zeros(gap, dtype=np.int16)))
        digest.update(signal.astype('<i2').tobytes()); total_samples += len(signal)
        chunks.append(features(signal, frontend))
        if len(prefix) < 2:
            prefix.append(signal)
    if len(prefix) != 2:
        raise ValueError('Need two utterances for continuity audit')
    np.testing.assert_array_equal(np.concatenate(chunks[:2]), features(np.concatenate(prefix)))
    data = np.concatenate(chunks)
    if len(data) != total_samples//160-2:
        raise ValueError('Unexpected continuous frontend cadence')
    first = rows[0]; relative = Path('human')/first['split']/(first['speaker']+'.npy')
    target = output/relative; target.parent.mkdir(parents=True, exist_ok=True); np.save(target, data)
    return dict(path=str(relative), split=first['split'], speaker=first['speaker'], language=first['language'],
                frames=len(data), files=len(rows), seconds=total_samples/16000,
                source_seconds=source_samples/16000, inserted_silence_seconds=(total_samples-source_samples)/16000,
                stream_pcm_sha256=digest.hexdigest(), feature_sha256=sha(target),
                first_two_chunk_continuity_checked=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    audit = json.loads((a.source/'audit.json').read_text())
    if sha(a.source/'manifest.jsonl') != audit['manifest_sha256']:
        raise ValueError('Changed audited corpus manifest')
    rows = [json.loads(x) for x in (a.source/'manifest.jsonl').read_text().splitlines()]
    groups, splits = defaultdict(list), {}
    for row in rows:
        if row['speaker'] in splits and splits[row['speaker']] != row['split']:
            raise ValueError('Speaker crosses splits')
        splits[row['speaker']] = row['split']; groups[row['speaker']].append(row)
    a.output.mkdir(parents=True)
    jobs = [(str(a.source), str(a.output), rows) for _, rows in sorted(groups.items())]
    result = []
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(worker, jobs):
            result.append(row); print(json.dumps(dict(speaker=row['speaker'], frames=row['frames'])), flush=True)
    (a.output/'human.json').write_text(json.dumps(result, indent=2)+'\n')
    report = dict(role='Waveform-derived continuous MicroFrontend cache, not classifier evidence',
                  source_manifest_sha256=audit['manifest_sha256'], source_audit_sha256=sha(a.source/'audit.json'),
                  source_sha256=sha(Path(__file__)), frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  index_sha256=sha(a.output/'human.json'), streams=len(result),
                  frames=sum(r['frames'] for r in result), split_hours={s:sum(r['source_seconds']/3600 for r in result if r['split']==s) for s in sorted(set(splits.values()))},
                  boundary_policy='200ms zero gap plus 0..159 alignment samples; continuous frontend state; exposure excludes all inserted silence',
                  chunk_continuity_checks_passed=True, classifiers_scored=False,
                  independent_final_audio_accessed=False)
    (a.output/'cache.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
