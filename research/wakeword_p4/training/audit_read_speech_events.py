"""Locate validation events in verified read-speech source timelines, without relabeling."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from evaluate import Detector, event_indices, smooth
from features import features as frontend_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'features', 'evaluation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--context-seconds', type=float, default=1.7,
                   help='Conservative preceding interval; not an exact acoustic/phonetic boundary')
    p.add_argument('--replay-model', type=Path,
                   help='Optional same-model original utterance controls at 0/10/20ms zero leads')
    a = p.parse_args()
    if a.output.exists() or not np.isfinite(a.context_seconds) or a.context_seconds <= 0:
        p.error('Fresh output and positive finite context required')
    cache = json.loads((a.features/'cache.json').read_text())
    report = json.loads((a.evaluation/'report.json').read_text())
    if (report.get('split') != 'validation' or report.get('independent_final_scored') is not False
            or report.get('development_regression_scored') is not False
            or cache['source_manifest_sha256'] != sha(a.source/'manifest.jsonl')
            or cache['source_audit_sha256'] != sha(a.source/'audit.json')
            or cache['index_sha256'] != sha(a.features/'human.json')
            or report['feature_index_sha256'] != cache['index_sha256']
            or report['feature_cache_sha256'] != sha(a.features/'cache.json')
            or not cache['boundary_policy'].startswith('200ms zero gap')):
        raise ValueError('Need matching default-gap read-speech validation provenance')
    sources = [json.loads(line) for line in (a.source/'manifest.jsonl').read_text().splitlines()]
    detector = None
    if a.replay_model:
        if sha(a.replay_model) != report['model_sha256']:
            raise ValueError('Replay model differs from scored model')
        detector = Detector(a.replay_model)
    features = {r['speaker']: r for r in json.loads((a.features/'human.json').read_text())
                if r['split'] == 'validation'}
    results = []
    for measurement in report['speakers']:
        if not measurement['false_events']:
            continue
        speaker = measurement['speaker']
        feature = features[speaker]
        scores_path = a.evaluation/(speaker+'-scores.npy')
        if sha(scores_path) != measurement['score_sha256']:
            raise ValueError('Changed score trace')
        scores = np.load(scores_path, allow_pickle=False)
        indices = event_indices(scores, report['threshold'])
        times = [float((3*(i+report['sliding_window']-1)+5)*.01) for i in indices]
        if times != measurement['event_seconds'] or len(times) != measurement['false_events']:
            raise ValueError('Event recount differs')
        selected = sorted([r for r in sources if r['speaker'] == speaker and r['split'] == 'validation'],
                          key=lambda r: r['path'])
        if not selected or len(selected) != feature['files']:
            raise ValueError('Incomplete source timeline')
        at, original_samples, digest, segments = 0, 0, hashlib.sha256(), []
        for row in selected:
            path = (a.source/row['path']).resolve()
            if not path.is_relative_to(a.source.resolve()) or sha(path) != row['sha256']:
                raise ValueError('Changed or escaped source WAV')
            pcm, rate = sf.read(path, dtype='int16')
            if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                    or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']):
                raise ValueError('Changed PCM')
            end = at+len(pcm)
            segments.append(dict(path=row['path'], text=row['text'], source_sha256=row['sha256'],
                                 start_sample=at, end_sample=end))
            gap = 3200+(-len(pcm)-3200) % 160
            digest.update(pcm.astype('<i2').tobytes())
            digest.update(np.zeros(gap, dtype='<i2').tobytes())
            at = end+gap
            original_samples += len(pcm)
        if (digest.hexdigest() != feature['stream_pcm_sha256'] or at/16000 != feature['seconds']
                or original_samples/16000 != feature['source_seconds']):
            raise ValueError('Reconstructed timeline differs from scored frontend source')
        for time in times:
            hi = round(time*16000)
            lo = max(0, hi-round(a.context_seconds*16000))
            overlap = [r for r in segments if r['start_sample'] < hi and r['end_sample'] > lo]
            entire = [r for r in overlap if r['start_sample'] <= lo and hi <= r['end_sample']]
            controls = []
            if detector is not None:
                for row in overlap:
                    pcm, rate = sf.read(a.source/row['path'], dtype='int16')
                    for lead in (0, 160, 320):
                        signal = np.concatenate((np.zeros(lead, dtype=np.int16), pcm))
                        # Same frontend cadence, with only final alignment padding.
                        signal = np.pad(signal, (0, (-len(signal)) % 160))
                        data = frontend_features(signal)
                        detector.reset()
                        values = smooth(detector.predict(data), report['sliding_window'])
                        detector.reset()
                        np.testing.assert_array_equal(values, smooth(detector.predict(data), report['sliding_window']))
                        events = event_indices(values, report['threshold'])
                        controls.append(dict(path=row['path'], zero_lead_samples=lead,
                                             event_seconds_relative_to_utterance=[float((3*(i+report['sliding_window']-1)+5)*.01-lead/16000) for i in events],
                                             peak=float(np.max(values)), reset_repeatability_passed=True,
                                             input_pcm_sha256=hashlib.sha256(signal.astype('<i2').tobytes()).hexdigest()))
            results.append(dict(speaker=speaker, event_seconds=time, context_start_sample=lo,
                                context_end_sample=hi, overlapping_sources=overlap,
                                entire_context_inside_one_original_utterance=bool(entire),
                                isolated_utterance_controls=controls,
                                reconstructed_stream_pcm_sha256=digest.hexdigest()))
    result = dict(role='Read-speech validation event source attribution, not phonetic labels or acceptance',
                  source_sha256=sha(Path(__file__)), source_manifest_sha256=sha(a.source/'manifest.jsonl'),
                  feature_cache_sha256=sha(a.features/'cache.json'), report_sha256=sha(a.evaluation/'report.json'),
                  detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  model_sha256=report['model_sha256'], threshold=report['threshold'],
                  context_seconds=a.context_seconds, events=results,
                  all_reported_events_located=len(results) == report['false_events'],
                  independent_final_audio_accessed=False, labels_or_thresholds_changed=False,
                  limits='Conservative preceding context, not exact model attribution or phonetic truth. Source transcripts cannot establish what caused an activation. Original utterance gaps and all events retained. Optional isolated utterance controls reset frontend/model and vary zero lead, so are distinct acoustic/state/phase conditions, not guaranteed reproductions of original continuous scores. Reconstructs only validation speakers with existing reported events; no training or final audio inference.')
    if not result['all_reported_events_located']:
        raise ValueError('Incomplete event coverage')
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
