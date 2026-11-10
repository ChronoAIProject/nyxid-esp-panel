"""Measure dependence on the terminal acoustic tail in known training WAVs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_stream_equal_mean import checked, steps
from detector_policy import smooth
from evaluate import Detector
from features import features


def truncate(pcm, end, fade):
    if end == len(pcm):
        return pcm.copy()
    if not fade < end < len(pcm):
        raise ValueError('Invalid truncation bounds')
    result = pcm.copy()
    result[end - fade:end] = np.rint(result[end - fade:end] * np.linspace(1, 0, fade))
    result[end:] = 0
    return result


def ready_samples(values, threshold, prime):
    # The first three frontend frames fit in the second 480-sample firmware
    # chunk. A valid five-score average first becomes visible in chunk six.
    return [(step + 6) * 480 - prime for step in steps(values, threshold)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('plan', 'ctc-rows', 'jobs', 'models', 'output'):
        p.add_argument('--' + arg, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report')
    plan = json.loads(a.plan.read_bytes())
    if plan['event_policy'] != {'smooth_steps': 5, 'step_samples': 480, 'refractory_steps': 67}:
        raise ValueError('Unreviewed event geometry')
    rows = [json.loads(x) for x in checked(a.ctc_rows, plan['ctc_rows_sha256']).splitlines()]
    jobs = json.loads(checked(a.jobs, plan['tts_jobs_sha256']))
    by_id = {job['id']: job for job in jobs}
    if len(jobs) != len(by_id):
        raise ValueError('Duplicate source job')
    heldout = {job['source_sha256'] for job in jobs if job['split'] != 'training'}
    models = {}
    for name, binding in plan['models'].items():
        directory = {'control': 'model-control', 'focus': 'model-focus-export-retry'}[name]
        model = a.models / directory / 'hey_nyx.tflite'
        checked(model, binding['model_sha256'])
        calibration = json.loads(checked(a.models / f'validation-{name}' / 'threshold.json',
                                         binding['calibration_sha256']))
        if (calibration['model_sha256'] != binding['model_sha256']
                or calibration['selected']['threshold'] != binding['threshold']
                or calibration.get('smoothing_steps', 5) != 5 or calibration['viable_operating_point']):
            raise ValueError('Changed fallback model or event policy')
        models[name] = (Detector(model), binding['threshold'])
    audio = plan['audio']
    prime, tail = audio['leading_silence_samples'], audio['trailing_silence_samples']
    if audio['sample_rate'] != 16000 or prime % 480:
        raise ValueError('Invalid PCM or prime geometry')
    outcomes = []
    seen = set()
    summary = {k: defaultdict(Counter) for k in models}
    for row in rows:
        spans = row['spans']
        if (len(spans) < 2 or [v['token'] for v in spans[-2:]] != ['k', 's']
                or spans[-1]['mean_argmax_probability'] < .5):
            continue
        job = by_id[row['variant0_id']]
        sha = row['source_sha256']
        if ((job['split'], job['kind'], job['label'], job['variant'], job['source_sha256'], job['engine'])
                != ('training', 'tts', 1, 0, sha, row['engine']) or sha in heldout or sha in seen):
            raise ValueError('Source is not a unique original training positive')
        seen.add(sha)
        path = Path(job['source'])
        checked(path, sha)
        pcm, rate = sf.read(path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']:
            raise ValueError('Original WAV geometry differs')
        s_start, s_end = spans[-1]['feature_input_support_samples']
        cut_before = s_start - audio['pre_support_margin_samples']
        cut_after = min(len(pcm), s_end + audio['post_support_margin_samples'])
        fade = audio['fade_samples']
        variants = {'intact': pcm, 'before_s_support': truncate(pcm, cut_before, fade),
                    'after_s_support': truncate(pcm, cut_after, fade)}
        inputs = {k: features(np.pad(v, (prime, tail + (-(len(v) + tail)) % 480)))
                  for k, v in variants.items()}
        result = {'source_sha256': sha, 'engine': row['engine'],
                  'ctc_support_samples': [s_start, s_end], 'cuts_samples': [cut_before, cut_after],
                  'last_pcm_sample': len(pcm), 'events_from_source_start_samples': {}}
        for name, (detector, threshold) in models.items():
            events = {}
            for key, data in inputs.items():
                detector.reset()
                scores = smooth(detector.predict(data), 5)
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite detector score')
                events[key] = ready_samples(scores, float(np.float32(threshold)), prime)
            for key, cut in (('before_s_support', cut_before), ('after_s_support', cut_after)):
                if [v for v in events[key] if v <= cut - fade] != [
                        v for v in events['intact'] if v <= cut - fade]:
                    raise ValueError('A suffix edit changed a causal prefix event')
            result['events_from_source_start_samples'][name] = events
            for engine in ('all', row['engine']):
                count = summary[name][engine]
                hits = {k: any(t >= 0 for t in v) for k, v in events.items()}
                count.update({
                    'sources': 1, 'intact_hit': hits['intact'],
                    'before_s_cut_hit': hits['before_s_support'], 'after_s_cut_hit': hits['after_s_support'],
                    'intact_and_before_s_cut_hit': hits['intact'] and hits['before_s_support'],
                    'intact_and_after_s_cut_hit': hits['intact'] and hits['after_s_support'],
                    'intact_before_ctc_support_by_40ms': any(0 <= t <= cut_before for t in events['intact']),
                    'any_prime_activation': any(t < 0 for v in events.values() for t in v),
                })
        outcomes.append(result)
    if not outcomes:
        raise ValueError('No frozen sources selected')
    report = {'scope': __doc__, 'plan_sha256': hashlib.sha256(a.plan.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'summary': summary, 'rows': outcomes, 'limits': plan['limits']}
    with a.output.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v['all'] for k, v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
