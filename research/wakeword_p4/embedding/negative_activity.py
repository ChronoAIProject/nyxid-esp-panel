"""Training-only TTS-negative endpoint intervals, never a runtime VAD.

Preserve source audio and all original feature arrays. The interval is an
uncertain energy-span sampling aid, not phonetic truth or an assertion that
histories ending outside it contain no speech.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms_span(pcm):
    if pcm.ndim != 1 or not len(pcm):
        raise ValueError('Expected nonempty mono source')
    padded = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
    rms = np.sqrt(np.mean(padded.reshape(-1, 160) ** 2, axis=1))
    active = np.flatnonzero(rms >= max(1., float(rms.max()) * .01))
    if not len(active):
        raise ValueError('Silent source has no sampling interval')
    return int(active[0]) * .01, min(len(pcm), (int(active[-1]) + 1) * 160) / 16000


def interval_map(rows, records, minimum_end):
    if not records:
        return {}
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Ambiguous feature identity')
    result = {}
    for record in records:
        identity = record['id']
        if identity in result or identity not in by_id:
            raise ValueError('Duplicate or unknown negative interval identity')
        row = by_id[identity]
        if (row['split'], row['kind'], row['label']) != ('training', 'tts', 0):
            raise ValueError('Intervals require training-only TTS negatives')
        if (any(row[key] != record[key] for key in
                ('source_sha256', 'features_sha256', 'engine', 'variant', 'frames'))
                or row['_feature_directory'] != record['feature_directory']):
            raise ValueError('Negative interval source or feature identity differs')
        low, high = record['minimum_end'], record['maximum_end']
        if type(low) is not int or type(high) is not int or not minimum_end <= low <= high <= row['frames']:
            raise ValueError('Negative interval exceeds warmed feature bounds')
        result[identity] = (low, high)
    expected = {row['id'] for row in rows if row['kind'] == 'tts' and not row['label']}
    if set(result) != expected:
        raise ValueError('Negative intervals must cover every original training TTS-negative view')
    return result


def prepare(args):
    if args.output.exists() or args.report.exists():
        raise ValueError('Use fresh manifest and report paths')
    plan = json.loads(args.plan.read_text())
    if sha(Path(__file__)) != plan['preparation_script_sha256']:
        raise ValueError('Frozen interval-preparation source differs')
    rows, jobs, heldout, positive_sources = [], {}, set(), set()
    for directory, binding in plan['feature_inputs'].items():
        root = args.data_root / directory
        for name, expected in binding.items():
            if sha(root / name) != expected:
                raise ValueError('Frozen feature/source input differs')
        source_jobs = json.loads((root / 'jobs-private.json').read_text())
        if len({row['id'] for row in source_jobs}) != len(source_jobs):
            raise ValueError('Duplicate original source jobs')
        jobs[directory] = {row['id']: row for row in source_jobs}
        for line in (root / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'training':
                heldout.add(row['source_sha256'])
            elif row['kind'] == 'tts' and not row['label']:
                rows.append(row | {'_feature_directory': directory})
            elif row['label']:
                positive_sources.add(row['source_sha256'])
    spans, records = {}, []
    for row in rows:
        source = row['source_sha256']
        job = jobs[row['_feature_directory']][row['id']]
        if (source in heldout or source in positive_sources or any(row[key] != job[key] for key in
                ('source_sha256', 'engine', 'variant', 'split', 'kind', 'label'))):
            raise ValueError('Original negative source role or partition differs')
        if source not in spans:
            path = Path(job['source'])
            if sha(path) != source:
                raise ValueError('Original source WAV differs')
            with wave.open(str(path), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError('Expected unchanged 16kHz mono PCM16')
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
            if abs(len(pcm) / 16000 - row['source_seconds']) > 1e-9:
                raise ValueError('Source duration or fixed feature timebase differs')
            spans[source] = rms_span(pcm)
        start, stop = spans[source]
        low, high = int(np.ceil((2.4 + start + .03) / .08)), int(np.ceil((2.4 + stop + .32) / .08))
        record = {key: row[key] for key in ('id', 'source_sha256', 'features_sha256', 'engine', 'variant', 'frames')}
        records.append(record | {'feature_directory': row['_feature_directory'],
                                  'minimum_end': low, 'maximum_end': high})
    interval_map(rows, records, 25)
    manifest = {'scope': __doc__, 'preparation_plan_sha256': sha(args.plan), 'records': records,
                'all_sources_training_only': True, 'heldout_source_hash_disjoint': True,
                'source_audio_unchanged': True, 'audio_speed_changed': False,
                'speaker_playback': False, 'microphone_access': False,
                'source_recordings': len(spans), 'accepted': False}
    with args.output.open('x') as out:
        out.write(json.dumps(manifest, indent=2) + '\n')
    groups = defaultdict(list)
    for record in records:
        groups[record['engine']].append((record['maximum_end'] - record['minimum_end'] + 1) /
                                        (record['frames'] - 25 + 1))
    cohorts = [{'engine': engine, 'feature_views': len(values),
                 'uniform_endpoint_fraction_in_interval': float(np.mean(values)),
                 'quarter_focused_expected_fraction': .25 + .75 * float(np.mean(values))}
                for engine, values in sorted(groups.items())]
    report = {key: value for key, value in manifest.items() if key != 'records'}
    report.update(manifest_sha256=sha(args.output), feature_views=len(records), cohorts=cohorts,
        engine_macro_uniform_endpoint_fraction=float(np.mean([row['uniform_endpoint_fraction_in_interval'] for row in cohorts])),
        limits=['Endpoint membership is not speech occupancy: original embeddings and 16-frame histories retain earlier speech.',
                'A broad first/last-RMS interval may include internal silence and omits uncertain phonetic/reverb boundaries.',
                'The theoretical fractions follow the original equal-engine/uniform-view sampler, not independent speaker exposure.',
                'No model is loaded, no labels/features/waveforms changed, no detector threshold or validation/final inference.'])
    with args.report.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'plan', 'output', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(parser.parse_args())
