"""Authenticate full streaming traces and report synthetic presentation metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(path, digest, samples):
    assert sha(path) == digest, 'Changed score trace'
    scores = np.load(path, allow_pickle=False)
    # MicroFrontend: 30ms window, 10ms hop; model consumes3 frames; mean5.
    expected = ((samples+159)//160-2)//3-4
    assert scores.shape == (expected,) and np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    return scores


def load(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((root/'report.json').read_text())
    assert not manifest['independent_final_audio_accessed'] and not report['final_data_used']
    assert report['manifest_sha256'] == sha(manifest_path)
    assert [item['record'] for item in report['records']] == manifest['records']
    clips = []
    for i, item in enumerate(report['records']):
        row = item['record']
        assert row['role'] == 'development-paired-validation' and row['split'] == 'validation'
        scores = trace(root/f'{i}-scores.npy', item['nyx']['score_sha256'], row['samples'])
        times = event_times(scores, report['threshold'])
        assert times == item['nyx']['event_seconds'], 'Saved events differ from trace'
        clips.append((row, scores, item['hiesp']['event_seconds']))
    return report, clips


def event_times(scores, threshold):
    return [float((3*(i+4)+5)*.01) for i in event_indices(scores, threshold)]


def metrics(clips, threshold, engine='nyx'):
    assert engine in ('nyx', 'hiesp')
    target = 'hey_nyx' if engine == 'nyx' else 'hi_esp'
    groups = defaultdict(lambda: dict(positives=0, triggered=0))
    negatives = {k: dict(clips=0, false_activation_clips=0, events=0)
                 for k in ('common', 'cross-target')}
    rows = []
    latencies, native_latencies = [], []
    pre_generation = late = 0
    for row, scores, official in clips:
        times = event_times(scores, threshold) if engine == 'nyx' else official
        assert all(np.isfinite(t) and 0 <= t <= row['samples']/16000+.1 for t in times)
        assert times == sorted(times)
        lead, end = row['lead_samples']/16000, row['keyword_end']/16000
        native_end = row.get('native_generation_end', row['keyword_end'])/16000
        record = dict(path=row['path'], voice=row['voice'], variant=row.get('variant', 0),
                      intended_phrase=row['intended_phrase'], event_seconds=times)
        if row['target'] == target:
            hits = [t for t in times if lead <= t <= end+.8]
            group = groups[(row['voice'], row.get('variant', 0))]
            group['positives'] += 1
            group['triggered'] += bool(hits)
            pre_generation += sum(t < lead for t in times)
            late += sum(t > end+.8 for t in times)
            latency = (hits[0]-end)*1000 if hits else None
            native_latency = (hits[0]-native_end)*1000 if hits else None
            if hits:
                latencies.append(latency)
                native_latencies.append(native_latency)
            record.update(presentation_trigger=bool(hits),
                          first_trigger_relative_generation_and_rir_end_ms=latency,
                          first_trigger_relative_native_generation_end_ms=native_latency)
        else:
            kind = 'common' if row['target'] is None else 'cross-target'
            negatives[kind]['clips'] += 1
            negatives[kind]['false_activation_clips'] += bool(times)
            negatives[kind]['events'] += len(times)
            record['negative_kind'] = kind
        rows.append(record)
    assert groups
    return dict(positive_clips=sum(g['positives'] for g in groups.values()),
        presentation_trigger_clips=sum(g['triggered'] for g in groups.values()),
        equal_voice_condition_macro=float(np.mean([g['triggered']/g['positives'] for g in groups.values()])),
        voice_conditions=[dict(voice=k[0], variant=k[1], **v) for k, v in sorted(groups.items())],
        pre_generation_positive_events=pre_generation, late_positive_events=late,
        negatives=negatives,
        first_trigger_relative_generation_and_rir_end_ms_median=float(np.median(latencies)) if latencies else None,
        first_trigger_relative_generation_and_rir_end_ms_p95=float(np.percentile(latencies, 95)) if latencies else None,
        first_trigger_relative_native_generation_end_ms_median=float(np.median(native_latencies)) if native_latencies else None,
        first_trigger_relative_native_generation_end_ms_p95=float(np.percentile(native_latencies, 95)) if native_latencies else None,
        records=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh report required')
    report, clips = load(a.scores, a.manifest)
    result = dict(model_sha256=report['model_sha256'], threshold=report['threshold'],
        source_report_sha256=sha(a.scores/'report.json'), manifest_sha256=sha(a.manifest),
        source_sha256=sha(Path(__file__)),
        summaries={engine: metrics(clips, report['threshold'], engine) for engine in ('nyx', 'hiesp')},
        final_data_used=False,
        limits='Development only. Lead..full generation/RIR end+800ms is a synthetic presentation interval, not verified complete-phoneme detection. Negative latency is valid relative to generated silence/tail, not proof of predictive detection. Augmentations/voices correlated; no human or hardware latency claim. Official threshold independently frozen .63, not matched population FAR.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'records'} for k, v in result['summaries'].items()}), flush=True)


if __name__ == '__main__':
    main()
