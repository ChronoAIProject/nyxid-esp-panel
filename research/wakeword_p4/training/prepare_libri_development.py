"""Audit official dev-other and prepare speaker-disjoint development negatives.

Never reads a model or the independent-final audio. Existing final manifest is
read only for speaker IDs; the new entire corpus is validation, not final.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tarfile
import numpy as np
import soundfile as sf
from pymicro_features import MicroFrontend
from features import features

EXPECTED_MD5 = 'c8d0bcc9cca99d4f8b62fcc847357931'
URL = 'https://www.openslr.org/resources/12/dev-other.tar.gz'


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def worker(job):
    corpus, output, speaker = job
    corpus, output = Path(corpus), Path(output)
    frontend = MicroFrontend()
    chunks, first, sources = [], [], []
    samples, total, pcm_hash = 0, 0, hashlib.sha256()
    for path in sorted((corpus/str(speaker)).rglob('*.flac')):
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or not len(pcm):
            raise ValueError('Invalid source audio')
        samples += len(pcm)
        sources.append(dict(path=str(path.relative_to(corpus)), sha256=sha(path), samples=len(pcm),
                            pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
        signal = np.pad(pcm, (0, 3200+(-len(pcm)-3200) % 160))
        pcm_hash.update(signal.astype('<i2').tobytes()); total += len(signal)
        chunks.append(features(signal, frontend))
        if len(first) < 2:
            first.append(signal)
    if len(first) != 2:
        raise ValueError('Need two utterances for continuity audit')
    np.testing.assert_array_equal(np.concatenate(chunks[:2]), features(np.concatenate(first)))
    data = np.concatenate(chunks)
    if len(data) != total//160-2:
        raise ValueError('Unexpected frontend cadence')
    relative = Path('human/validation')/f'{speaker}.npy'
    target = output/relative; target.parent.mkdir(parents=True, exist_ok=True); np.save(target, data)
    return dict(speaker=f'libri_dev_other_{speaker}', original_speaker_id=speaker, split='validation',
                category='english-hard-read', language='en', path=str(relative), frames=len(data),
                seconds=total/16000, source_seconds=samples/16000,
                inserted_silence_seconds=(total-samples)/16000, files=len(sources),
                stream_pcm_sha256=pcm_hash.hexdigest(), feature_sha256=sha(target),
                first_two_chunk_continuity_checked=True, sources=sources)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--development-human', type=Path, required=True)
    p.add_argument('--final-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists() or (a.work/'LibriSpeech/dev-other').exists():
        p.error('Use fresh output and extraction directories')
    if sha(a.archive, 'md5') != EXPECTED_MD5:
        raise ValueError('Official MD5 mismatch')
    a.work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive) as archive:
        members = archive.getmembers()
        for member in members:
            target = (a.work/member.name).resolve()
            if not target.is_relative_to(a.work.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Archive contains escaped path or unsupported link/device')
        archive.extractall(a.work, members=members)
    corpus = a.work/'LibriSpeech/dev-other'
    speakers = sorted(int(x.name) for x in corpus.iterdir() if x.is_dir())
    previous = {int(r['speaker']) for r in json.loads(a.development_human.read_text())
                if str(r['speaker']).isdigit()}
    frozen = json.loads(a.final_manifest.read_text())
    final_ids = {int(r['speaker']) for r in frozen['records']}
    if previous.intersection(speakers) or final_ids.intersection(speakers):
        raise ValueError('Speaker crosses existing development or independent final')
    transcripts = []
    for path in sorted(corpus.rglob('*.trans.txt')):
        if re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks)\b', path.read_text(), re.I):
            raise ValueError('Potential target phrase needs review')
        transcripts.append(dict(path=str(path.relative_to(corpus)), sha256=sha(path)))
    a.output.mkdir(parents=True)
    jobs = [(str(corpus), str(a.output), speaker) for speaker in speakers]
    records = []
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(worker, jobs):
            records.append(row)
            print(json.dumps(dict(speaker=row['speaker'], completed=len(records), planned=len(jobs))), flush=True)
    (a.output/'human.json').write_text(json.dumps(records, indent=2)+'\n')
    report = dict(role='Audited dev-other development negative features, not recognition evidence',
                  url=URL, license='CC-BY-4.0', official_md5=EXPECTED_MD5, archive_sha256=sha(a.archive),
                  source_sha256=sha(Path(__file__)), frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  index_sha256=sha(a.output/'human.json'), speakers=speakers,
                  existing_development_human_sha256=sha(a.development_human),
                  final_manifest_sha256=sha(a.final_manifest), speaker_disjoint=True,
                  source_hours=sum(r['source_seconds'] for r in records)/3600,
                  transcripts=transcripts, scores_computed=False, independent_final_audio_accessed=False,
                  limits='Development read speech only. Transcript literal screen is not a phonetic audit. Speaker utterances concatenated with 200ms silence and alignment padding, excluded from exposure; no real continuous conversation or positive recall evidence.')
    (a.output/'cache.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'transcripts'}), flush=True)


if __name__ == '__main__':
    main()
