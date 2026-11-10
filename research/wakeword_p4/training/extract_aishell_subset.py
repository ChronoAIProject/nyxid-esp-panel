"""Audit and extract only pinned, original-train AISHELL speaker archives."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import soundfile as sf


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('Use a fresh output directory; preserve partial audits')
source = json.loads((a.source/'download.json').read_text())
if not source['all_archive_hashes_verified']:
    raise ValueError('Incomplete download audit')
for name, digest in source['metadata_sha256'].items():
    if sha(a.source/name) != digest:
        raise ValueError('Changed source metadata')
transcripts = {}
for line in (a.source/'data_aishell/transcript/aishell_transcript_v0.8.txt').read_text().splitlines():
    fields = line.split(maxsplit=1)
    if len(fields) == 2:
        if fields[0] in transcripts:
            raise ValueError('Duplicate transcript identity')
        transcripts[fields[0]] = fields[1]
a.output.mkdir(parents=True)
rows, speakers, pcm_seen, paths = [], {}, {}, set()
hours, counts, quarantined = Counter(), Counter(), []
for archive in source['archives']:
    filename = a.source/archive['path']
    if sha(filename) != archive['sha256'] or filename.stat().st_size != archive['size']:
        raise ValueError('Changed archive')
    speaker = archive['speaker']
    split = 'testing' if archive['split'] == 'testing_regression' else archive['split']
    if speaker in speakers:
        raise ValueError('Speaker appears twice in plan')
    speakers[speaker] = split
    local_count = 0
    with tarfile.open(filename, mode='r|gz') as stream:
        for member in stream:
            if member.isdir():
                continue
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 3 or parts[:2] != ('train', speaker)
                    or not parts[2].endswith('.wav') or member.size > 4 << 20):
                raise ValueError(f'Unexpected original partition or archive member: {member.name}')
            identity = Path(parts[2]).stem
            if not identity.startswith('BAC009'+speaker+'W'):
                raise ValueError('Audio/transcript speaker identity mismatch')
            payload = stream.extractfile(member).read()
            info = sf.info(io.BytesIO(payload))
            if info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16':
                raise ValueError('Unexpected PCM format')
            if info.frames <= 0:
                quarantined.append(dict(identity=identity, speaker=speaker, split=split,
                                        seconds=0, sha256=hashlib.sha256(payload).hexdigest(),
                                        reason='Zero-frame source WAV; no usable audio'))
                continue
            pcm, _ = sf.read(io.BytesIO(payload), dtype='int16')
            pcm_digest = hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
            if pcm_digest in pcm_seen:
                raise ValueError('Duplicate PCM in selected corpus')
            pcm_seen[pcm_digest] = split
            if identity not in transcripts:
                quarantined.append(dict(identity=identity, speaker=speaker, split=split,
                                        seconds=info.frames/16000, sha256=hashlib.sha256(payload).hexdigest(),
                                        reason='Source WAV lacks reference transcript; exclude without relabeling'))
                continue
            text = transcripts[identity]
            compact = ''.join(text.split())
            if re.search(r'\bhey\s+(nyx|nix|nicks)\b', text, re.I) or any(
                    x in compact for x in ['嘿尼克斯', '嗨尼克斯']):
                quarantined.append(dict(identity=identity, reason='Literal target candidate requires review'))
                continue
            path = Path('wav')/split/speaker/parts[2]
            if str(path) in paths:
                raise ValueError('Duplicate output path')
            paths.add(str(path)); target = a.output/path
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
            seconds = info.frames/16000
            rows.append(dict(path=str(path), split=split, speaker='aishell_'+speaker,
                             original_partition='train', language='zh', samples=info.frames,
                             seconds=seconds, sha256=hashlib.sha256(payload).hexdigest(),
                             pcm_sha256=pcm_digest, transcript_id=identity, text=text))
            counts[split] += 1; hours[split] += seconds/3600; local_count += 1
    if not local_count:
        raise ValueError('Empty selected speaker')
    print(json.dumps(dict(speaker=speaker, split=split, utterances=local_count)), flush=True)
(a.output/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
report = dict(role='Chinese read-speech development corpus audit; no model inference',
              source_download_sha256=sha(a.source/'download.json'), repo=source['repo'], revision=source['revision'],
              source_sha256=sha(Path(__file__)), source_license=source['license'],
              speakers_by_split=dict(Counter(speakers.values())), utterances_by_split=dict(counts),
              original_hours_by_split=dict(hours), source_partition_verified='train',
              pcm_format='16000 Hz mono PCM16', pcm_duplicates=0, retained_transcript_coverage_complete=True,
              raw_utterance_count=len(rows)+len(quarantined),
              manifest_sha256=sha(a.output/'manifest.jsonl'), quarantined=quarantined,
              classifiers_scored=False, independent_final_audio_accessed=False,
              limits='New custom speaker-disjoint development split within official train. Read Mandarin does not cover conversation/TV/far field. Transcript screening is not exhaustive acoustic verification. Cross-corpus physical speaker identity is not established.')
(a.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
