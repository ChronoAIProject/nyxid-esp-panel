"""Locate the paired-phrase experiment's false wake in retained development PCM."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np
import soundfile as sf

from evaluate import Detector, events, smooth
from features import continuous_wav_features

RATE = 16000
SOURCE = '1183-133256-0018'
REF = .97
WIDTH = 5
# Inside the gap between the two clauses, based on the separate offline
# transcript alignment. Fades preserve the original stream clock.
SPLIT_BEGIN, SPLIT_END = 1.80, 1.88


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def question_onsets(index, corpus, experiment):
    """Screen other in-corpus question onsets from already scored streams."""
    matched = []
    for i, row in enumerate(index):
        if row['split'] != 'validation' or not row.get('sources') or 'utterance' not in row['sources'][0]:
            continue
        at, scores, chapters = 0, None, {}
        for source in row['sources']:
            uid = source['utterance']
            speaker, chapter, _ = uid.split('-')
            if chapter not in chapters:
                path = corpus/speaker/chapter/f'{speaker}-{chapter}.trans.txt'
                chapters[chapter] = dict(s.split(' ', 1) for s in path.read_text().splitlines())
            sentence = chapters[chapter][uid]
            if hashlib.sha256(sentence.encode()).hexdigest() != source['transcript_sha256']:
                raise ValueError('Original transcript changed')
            group = re.match(r'(HAVE YOU|DO YOU|ARE YOU|CAN YOU)\b', sentence)
            if group:
                if scores is None:
                    scores = np.load(experiment/'validation-phrase_aux'/f'ambient-validation-{i:03d}.npy')
                    ends = (3 * (np.arange(len(scores)) + WIDTH - 1) + 5) * 160
                # Keep the same context length for each onset or stop at the
                # utterance boundary when it is shorter.
                mask = (ends >= at) & (ends <= at + min(source['samples'], 53600))
                matched.append({'utterance': uid, 'prefix': group[1], 'maximum_score': float(scores[mask].max())})
            at += source['samples'] + 3200
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'joined', 'corpus', 'experiment', 'scratch', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    idx = json.loads((args.features/'human.json').read_bytes())
    index, row = next((i, r) for i, r in enumerate(idx) if r.get('speaker') == '1183')
    if row['split'] != 'validation' or sha(args.joined) != row['pcm_sha256']:
        raise ValueError('Expected the original joined validation WAV')
    pcm, rate = sf.read(args.joined, dtype='int16')
    if rate != RATE or pcm.ndim != 1:
        raise ValueError('Expected mono 16 kHz PCM')
    origin = 0
    for source in row['sources']:
        if source['utterance'] == SOURCE:
            break
        origin += source['samples'] + 3200
    else:
        raise ValueError('Missing source utterance')
    last = origin + source['samples']
    folder = args.corpus/'1183'/'133256'
    flac = folder/f'{SOURCE}.flac'
    transcript = next(s.split(' ', 1)[1] for s in (folder/'1183-133256.trans.txt').read_text().splitlines()
                      if s.startswith(SOURCE + ' '))
    utterance, source_rate = sf.read(flac, dtype='int16')
    if (source_rate != RATE or sha(flac) != source['flac_sha256']
            or hashlib.sha256(transcript.encode()).hexdigest() != source['transcript_sha256']
            or not np.array_equal(utterance, pcm[origin:last])):
        raise ValueError('Source bytes do not match the indexed validation audio')

    args.scratch.mkdir(exist_ok=False)
    # Preserve the entire preceding stream (noise estimator and model state)
    # and stop after this utterance. All modifications occur at 16 kHz.
    prefix = pcm[:last + 3200].copy()
    a, b = (origin + round(t * RATE) for t in (SPLIT_BEGIN, SPLIT_END))
    fade = .5 - .5 * np.cos(np.linspace(0, np.pi, b-a))
    model = args.experiment/'model-phrase_aux'/'hey_nyx.tflite'
    calibration = json.loads((args.experiment/'validation-phrase_aux'/'threshold.json').read_bytes())
    if sha(model) != calibration['model_sha256'] or calibration['smoothing_steps'] != WIDTH:
        raise ValueError('Unexpected model or smoothing')
    key = f'ambient-validation-{index:03d}.npy'
    cached = np.load(args.experiment/'validation-phrase_aux'/key)
    control = np.load(args.experiment/'validation-phrase_control'/key)
    tracks = {}
    for variant in ('original', 'first_clause_only', 'second_clause_only', 'utterance_only'):
        edit = prefix.copy()
        if variant == 'first_clause_only':
            edit[a:b] = np.rint(edit[a:b] * (1-fade)).astype(np.int16)
            edit[b:] = 0
        elif variant == 'second_clause_only':
            edit[origin:a] = 0
            edit[a:b] = np.rint(edit[a:b] * fade).astype(np.int16)
        elif variant == 'utterance_only':
            edit[:origin] = 0
        path = args.scratch/f'{variant}.wav'
        with wave.open(str(path), 'wb') as out:
            out.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
            out.writeframes(edit.astype('<i2', copy=False).tobytes())
        raw, _ = continuous_wav_features(path)
        values = smooth(Detector(model).predict(raw), WIDTH)
        # Exclude the very last hop: this shortened stream may pad it whereas
        # the original continued with the next utterance.
        if variant == 'original' and not np.array_equal(values[:-1], cached[:len(values)-1]):
            raise ValueError('Original prefix does not reproduce the scored stream')
        samples = (3 * (np.arange(len(values)) + WIDTH - 1) + 5) * 160
        at = (samples >= origin) & (samples <= last)
        clip = values[at]
        hit = samples[at][clip >= REF]
        tracks[variant] = {'pcm_sha256': sha(path), 'maximum_score': float(clip.max()),
                           'events_at_0_97': events(clip, REF),
                           'first_above_sample': int(hit[0]) if len(hit) else None,
                           'last_above_sample': int(hit[-1]) if len(hit) else None}
    if tracks['original']['events_at_0_97'] != 1 or events(cached, REF) != 1:
        raise ValueError('Unexpected original event count')
    result = {'scope': 'Previously scored public LibriSpeech development stream; counterfactual diagnosis',
              'utterance': SOURCE, 'utterance_start_sample': origin,
              'source_sha256': source['flac_sha256'], 'joined_sha256': row['pcm_sha256'],
              'reference_model_sha256': calibration['model_sha256'],
              'reference_score_array_sha256': sha(args.experiment/'validation-phrase_aux'/key),
              'transcript': transcript, 'sample_rate': RATE, 'threshold': REF, 'smoothing_steps': WIDTH,
              'clause_fade_seconds_relative_to_utterance': [SPLIT_BEGIN, SPLIT_END],
              'control_maximum_on_full_stream': float(control.max()),
              'variants': tracks,
              'matched_question_onsets_in_scored_validation': question_onsets(idx, args.corpus, args.experiment),
              'limits': 'Automatic clause alignment and broad waveform ablation do not identify a particular phoneme. '
                        'No held-out source, speed change, playback, capture or device run.'}
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(tracks, indent=2))


if __name__ == '__main__':
    main()
