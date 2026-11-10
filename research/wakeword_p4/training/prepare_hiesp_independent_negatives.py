"""Reconstruct frozen test-negative PCM and verify every cached frontend frame."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from pymicro_features import MicroFrontend
import soundfile as sf
from features import features


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def worker(job):
    work, frozen, output, row, archive_hashes = job
    work, frozen, output = Path(work), Path(frozen), Path(output)
    prior = frozen/row['path']
    if sha(prior) != row['feature_sha256']:
        raise ValueError('Changed frozen feature cache')
    expected = np.load(prior, mmap_mode='r', allow_pickle=False)
    files = sorted((work/'LibriSpeech'/row['corpus']/str(row['speaker'])).rglob('*.flac'))
    if len(files) != row['files']:
        raise ValueError('Changed frozen speaker source count')
    frontend, at, original, frame_at = MicroFrontend(), 0, 0, 0
    sources, digest = [], hashlib.sha256()
    target = output/(row['corpus']+'-'+str(row['speaker'])+'.wav')
    with sf.SoundFile(target, 'w', samplerate=16000, channels=1, subtype='PCM_16') as stream:
        for path in files:
            relative = str(path.relative_to(work))
            if sha(path) != archive_hashes.get(relative):
                raise ValueError('Extracted FLAC differs from pinned official archive')
            pcm, rate = sf.read(path, dtype='int16')
            if rate != 16000 or pcm.ndim != 1:
                raise ValueError('Invalid original PCM format')
            # human_worker adds exactly 3200 samples, then features() pads each
            # utterance separately to 160-sample boundaries, sharing frontend.
            gap = 3200+(-len(pcm)-3200) % 160
            signal = np.pad(pcm, (0, gap))
            actual = features(signal, frontend)
            np.testing.assert_array_equal(actual, expected[frame_at:frame_at+len(actual)])
            frame_at += len(actual)
            stream.write(signal)
            digest.update(signal.astype('<i2').tobytes())
            sources.append(dict(path=relative, sha256=archive_hashes[relative], samples=len(pcm),
                                pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
            at += len(signal)
            original += len(pcm)
    if frame_at != row['frames'] or original/16000 != row['source_seconds']:
        raise ValueError('Changed frozen feature length or original exposure')
    return dict(path=target.name, role='independent-negative-final', split='testing',
                speaker=str(row['speaker']), corpus=row['corpus'], samples=at,
                source_seconds=original/16000, inserted_silence_and_alignment_seconds=(at-original)/16000,
                sha256=sha(target), pcm_sha256=digest.hexdigest(),
                prior_feature_sha256=row['feature_sha256'], all_frontend_frames_equal=True, sources=sources)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'frozen-manifest', 'nyx-preregistration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.workers <= 8:
        p.error('Fresh output and workers1..8 required')
    manifest = json.loads(a.frozen_manifest.read_text())
    freeze = json.loads(a.nyx_preregistration.read_text())
    if sha(a.frozen_manifest) != freeze['negative_manifest_sha256']:
        raise ValueError('Changed preregistered independent negatives')
    archive_hashes = {}
    for source in manifest['provenance']:
        archive = a.work/(source['corpus']+'.tar.gz')
        if sha(archive) != source['archive_sha256']:
            raise ValueError('Changed official test archive')
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if member.isfile() and member.name.endswith('.flac'):
                    stream = tar.extractfile(member)
                    archive_hashes[member.name] = hashlib.sha256(stream.read()).hexdigest()
    a.output.mkdir(parents=True)
    jobs = [(str(a.work), str(a.frozen_manifest.parent), str(a.output), row, archive_hashes)
            for row in manifest['records']]
    records = []
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(worker, jobs):
            records.append(row)
            print(json.dumps(dict(completed=len(records), planned=len(jobs))), flush=True)
    result = dict(role='Same frozen independent negative PCM for official Hi ESP, no new Nyx inference',
                  source_sha256=sha(Path(__file__)), frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  frozen_negative_manifest_sha256=sha(a.frozen_manifest),
                  nyx_preregistration_sha256=sha(a.nyx_preregistration),
                  utterance_concatenation=True, all_frontend_frames_equal=True,
                  original_source_hours=sum(r['source_seconds'] for r in records)/3600,
                  independent_negative_final_audio_accessed=True, independent_positive_final_audio_accessed=False,
                  records=records, limits='Pinned official archive FLAC bodies verified; all reconstructed frontend frames equal frozen Nyx inputs. Inserted 200ms gaps and per-utterance alignment padding reproduce the original cache, but are excluded from exposure. No new detector scores or threshold changes. Read speech only, not natural conversation.')
    (a.output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
