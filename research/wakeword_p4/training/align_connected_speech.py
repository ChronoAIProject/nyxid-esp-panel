"""Locate training-only near-word spans with offline, unprompted word timing."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

import av
from faster_whisper import WhisperModel
import numpy as np

from connected_speech_candidates import ALIASES, WORDS

MODEL_SHA = '1a5afae06a4db91c975c9a9d78be5cc110ee4ea022ad57d55492e4550e936b2a'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_pcm(path, expected):
    with av.open(str(path), mode='r') as container:
        parts = []
        for frame in container.decode(audio=0):
            if frame.sample_rate != 16000 or frame.format.name != 's16' or frame.layout.nb_channels != 1:
                raise ValueError('Expected original 16 kHz mono PCM16')
            parts.append(frame.to_ndarray().reshape(-1))
        pcm = np.concatenate(parts)
    if len(pcm) != expected:
        raise ValueError('Decoded utterance length differs from the source index')
    return pcm.astype(np.float32)/32768


def align(row, speech):
    tokens, times = [], []
    for word in speech:
        normalized = WORDS.findall(word.word.lower())
        if len(normalized) == 1:
            tokens.append(normalized[0])
            times.append((word.start, word.end))
        elif normalized:
            return []  # Timestamp boundaries are ambiguous inside this token.
    if ALIASES.search(' '.join(tokens)):
        return []
    original = row['words']
    whole = original == tokens
    blocks = SequenceMatcher(None, original, tokens, autojunk=False).get_matching_blocks()
    spans = []
    for hit in row['matches']:
        at, stop = hit['start_word'], hit['end_word']
        span = original[at:stop]
        if not whole and sum(tokens[i:i+len(span)] == span for i in range(len(tokens))) != 1:
            continue
        # Require adjacent words when present; the transcript determines what
        # was spoken, and ASR only locates a matching window for it.
        left, right = max(0, at-1), min(len(original), stop+1)
        block = next((b for b in blocks if b.a <= left and right <= b.a+b.size), None)
        if block is None:
            continue
        start_s, end_s = times[at+block.b-block.a][0], times[stop-1+block.b-block.a][1]
        start, end = round(start_s*16000), round(end_s*16000)
        if not (0 <= start < end <= row['source_samples'] and 1600 <= end-start <= 32000):
            continue
        spans.append({**hit, 'text': ' '.join(span), 'onset_samples': start, 'end_samples': end,
                      'stream_end_frame': (row['origin_samples']+end)//160-2})
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidates', 'corpus', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.model/'model.bin') != MODEL_SHA:
        raise ValueError('Use a fresh report and the pinned local ASR')
    candidate = json.loads(args.candidates.read_bytes())
    model = WhisperModel(str(args.model), device='cpu', compute_type='int8', cpu_threads=3, num_workers=1)
    output = []
    for i, row in enumerate(candidate['rows'], 1):
        speaker, chapter, _ = row['source'].split('-')
        src = args.corpus/speaker/chapter/(row['source']+'.flac')
        if sha(src) != row['source_sha256']:
            raise ValueError('Training source checksum mismatch')
        pcm = read_pcm(src, row['source_samples'])
        segments, _ = model.transcribe(pcm, language='en', beam_size=3, vad_filter=False,
            condition_on_previous_text=False, temperature=0, word_timestamps=True)
        spans = align(row, [w for seg in segments for w in seg.words])
        if spans:
            output.append({k: v for k, v in row.items() if k not in ('words', 'matches')} | {'spans': spans})
        if i % 25 == 0 or i == len(candidate['rows']):
            print(json.dumps({'scanned': i, 'retained': len(output)}), flush=True)
    report = {'scope': 'Training-only natural speech; word times estimated by unprompted offline ASR',
              'candidate_sha256': sha(args.candidates), 'index_sha256': candidate['index_sha256'],
              'asr_model_sha256': MODEL_SHA, 'sources': len(output), 'speakers': len({r['speaker'] for r in output}),
              'spans_by_distance': dict(Counter(m['distance'] for r in output for m in r['spans'])), 'rows': output}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
