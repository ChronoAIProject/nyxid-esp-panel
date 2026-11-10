"""Prepare checksum-verified train-clean-100 as training-only negative streams.

Final manifest is read only for identities; no final audio or classifier is read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import tarfile
from prepare_libri_development import sha, worker

EXPECTED_MD5 = '2a93770f6d5c6c964bc36631d331a522'
URL = 'https://www.openslr.org/resources/12/train-clean-100.tar.gz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'work', 'development-human', 'final-manifest', 'piper-config', 'speaker-metadata', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists() or (a.work/'LibriSpeech/train-clean-100').exists():
        p.error('Use fresh output and extraction directories')
    if a.workers < 1:
        p.error('workers must be positive')
    if sha(a.archive, 'md5') != EXPECTED_MD5:
        raise ValueError('Official archive MD5 mismatch')
    expected = set()
    for line in a.speaker_metadata.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(';'):
            continue
        fields = [x.strip() for x in line.split('|')]
        if fields[2] == 'train-clean-100':
            expected.add(int(fields[0]))
    if len(expected) != 251:
        raise ValueError('Expected official 251-speaker metadata')
    previous_rows = json.loads(a.development_human.read_text())
    previous = {int(r.get('original_speaker_id', r['speaker'])) for r in previous_rows
                if str(r.get('original_speaker_id', r['speaker'])).isdigit()}
    final = {int(r['speaker']) for r in json.loads(a.final_manifest.read_text())['records']}
    piper = {int(x) for x in json.loads(a.piper_config.read_text())['speaker_id_map']}
    if expected & (previous | final | piper):
        raise ValueError('Training corpus intersects existing human, final, or Piper identities')
    a.work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive) as archive:
        members = archive.getmembers()
        for member in members:
            target = (a.work/member.name).resolve()
            if not target.is_relative_to(a.work.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive path or member type')
        archive.extractall(a.work, members=members)
    corpus = a.work/'LibriSpeech/train-clean-100'
    speakers = sorted(int(x.name) for x in corpus.iterdir() if x.is_dir())
    if set(speakers) != expected:
        raise ValueError('Extracted speakers differ from official metadata')
    transcripts, transcript_ids = [], set()
    for path in sorted(corpus.rglob('*.trans.txt')):
        text = path.read_text()
        if re.search(r'\bhey\s+(?:nyx|nix|nicks|knicks|nick\s+s)\b', text, re.I):
            raise ValueError('Potential target in negative transcript requires review: '+str(path))
        for line in text.splitlines():
            utterance, sentence = line.split(maxsplit=1)
            if utterance in transcript_ids or not sentence.strip():
                raise ValueError('Duplicate or empty transcript')
            transcript_ids.add(utterance)
        transcripts.append(dict(path=str(path.relative_to(corpus)), sha256=sha(path)))
    audio_ids = [x.stem for x in corpus.rglob('*.flac')]
    if len(audio_ids) != len(set(audio_ids)) or set(audio_ids) != transcript_ids:
        raise ValueError('Audio/transcript coverage mismatch')
    a.output.mkdir(parents=True)
    # Join existing English stratum: do not introduce a seventh equal-weight class.
    jobs = [(str(corpus), str(a.output), speaker, 'training', 'libri_train_clean_100_', 'unspecified')
            for speaker in speakers]
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for row in pool.map(worker, jobs):
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), planned=len(jobs))), flush=True)
    (a.output/'human.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = dict(role='Training-only LibriSpeech negatives, not recognition evidence',
                  url=URL, license='CC-BY-4.0', official_md5=EXPECTED_MD5,
                  archive_sha256=sha(a.archive), index_sha256=sha(a.output/'human.json'),
                  source_sha256=sha(Path(__file__)),
                  worker_source_sha256=sha(Path(__file__).with_name('prepare_libri_development.py')),
                  frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  identity_inputs={name:sha(getattr(a,name)) for name in
                      ('development_human','final_manifest','piper_config','speaker_metadata')},
                  speakers=speakers, source_hours=sum(r['source_seconds'] for r in rows)/3600,
                  transcripts=transcripts, scores_computed=False, independent_final_audio_accessed=False,
                  speaker_disjoint=True, category='unspecified',
                  limits='Read speech only; transcript literal screen is not phonetic proof. 200ms gaps and alignment padding excluded from original exposure. Not natural conversation.')
    (a.output/'cache.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'transcripts'}), flush=True)


if __name__ == '__main__':
    main()
