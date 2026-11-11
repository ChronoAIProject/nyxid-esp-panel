"""Find naturally spoken near words in the existing LibriSpeech training split."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path
import re

DICTIONARY_SHA = '81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22'
TARGET = ('HH', 'EY', 'N', 'IH', 'K', 'S')
ALIASES = re.compile(r"\b(?:nyx|nix|nicks|nick['’]s|knicks|esp)\b|\bhi\s+e\s+s\s+p\b", re.I)
WORDS = re.compile(r"[a-z]+(?:'[a-z]+)*", re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dictionary(data):
    if digest(data) != DICTIONARY_SHA:
        raise ValueError('Use the pinned CMU dictionary')
    words = defaultdict(set)
    for line in data.decode().splitlines():
        if line and not line.startswith((';;', '#')):
            word, *phones = line.partition(' #')[0].split()
            word = word.split('(')[0]
            words[word].add(tuple(p.rstrip('012') for p in phones))
    return {word: sorted(choices) for word, choices in words.items()}


def distance(phones):
    prev = list(range(len(TARGET) + 1))
    for i, p in enumerate(phones, 1):
        row = [i]
        for j, q in enumerate(TARGET, 1):
            row.append(min(row[-1]+1, prev[j]+1, prev[j-1]+(p != q)))
        prev = row
    return prev[-1]


def find_matches(pronunciation):
    @lru_cache(maxsize=100000)
    def nearest(words):
        choices = [pronunciation.get(w, ()) for w in words]
        return min((distance(sum(seq, ())), sum(seq, ())) for seq in product(*choices)) if all(choices) else (99, ())

    def select(words):
        candidates = []
        for at in range(len(words)):
            for stop in range(at+1, min(at+3, len(words))+1):
                d, phones = nearest(words[at:stop])
                if d == 0:
                    return []  # No exact phonetic alias anywhere in this sentence.
                if d <= 2 or (d == 3 and len(phones) >= 4 and phones[-1] in ('S', 'Z')):
                    candidates.append((d, at, stop, phones))
        return candidates

    return select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dictionary', 'corpus', 'index', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    raw = args.index.read_bytes()
    base = args.manifest.read_bytes()
    corpus = json.loads(base)
    partitions = corpus['speaker_partitions']
    train = set(partitions['training'])
    excluded = set(partitions['validation']) | set(partitions['testing'])
    if train & excluded or corpus['archive_sha256'] != 'd4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2':
        raise ValueError('Unexpected LibriSpeech split or archive')
    index = json.loads(raw)
    nontrain = {r.get('speaker') for r in index if r['split'] != 'training'}
    if train & nontrain:
        raise ValueError('Training speaker found in a reserved split')
    find = find_matches(dictionary(args.dictionary.read_bytes()))
    seen, utterances = set(), 0
    rows = []
    for r in index:
        speaker = r.get('speaker')
        if r['split'] != 'training' or speaker not in train or not r.get('sources') or 'utterance' not in r['sources'][0]:
            continue
        if speaker in seen:
            raise ValueError('Duplicate training speaker stream')
        seen.add(speaker)
        transcripts, origin = {}, 0
        for source in r['sources']:
            uid = source['utterance']
            person, chapter, _ = uid.split('-')
            if person != speaker or not uid.replace('-', '').isdigit():
                raise ValueError('Unexpected source speaker or ID')
            folder = args.corpus/speaker/chapter
            if chapter not in transcripts:
                file = folder/f'{speaker}-{chapter}.trans.txt'
                transcripts[chapter] = dict(line.split(' ', 1) for line in file.read_text().splitlines())
            sentence = transcripts[chapter][uid]
            if digest(sentence.encode()) != source['transcript_sha256']:
                raise ValueError('Original transcript hash mismatch')
            words = tuple(w.lower() for w in WORDS.findall(sentence))
            if not ALIASES.search(sentence):
                hits = find(words)
                if hits:
                    rows.append({'source': uid, 'speaker': speaker, 'feature_path': r['path'],
                                 'feature_sha256': r['feature_sha256'], 'origin_samples': origin,
                                 'source_samples': source['samples'], 'source_sha256': source['flac_sha256'],
                                 'transcript_sha256': source['transcript_sha256'], 'words': words,
                                 'matches': [{'distance': d, 'start_word': start, 'end_word': end, 'phones': phones}
                                             for d, start, end, phones in hits]})
            origin += source['samples'] + 3200
            utterances += 1
    if seen != train or not rows:
        raise ValueError('Missing speaker streams or no connected-speech candidates')
    report = {'scope': 'Original training partition only; phonetic text candidates, not acoustic alignments',
              'dictionary_sha256': DICTIONARY_SHA, 'manifest_sha256': digest(base), 'index_sha256': digest(raw),
              'target_phones': TARGET, 'training_speakers': len(seen), 'training_utterances': utterances,
              'candidate_utterances': len(rows), 'candidate_speakers': len({r['speaker'] for r in rows}),
              'spans_by_distance': dict(Counter(m['distance'] for r in rows for m in r['matches'])), 'rows': rows}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
