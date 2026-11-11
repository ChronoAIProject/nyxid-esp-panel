"""Replay paired development WAVs through native Hi ESP and streaming Nyx INT8."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from features import features
from evaluate import Detector, smooth, event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(times, row, target):
    end = row['keyword_end']/16000
    if row['target'] != target:
        return dict(negative_kind='common' if row['target'] is None else 'cross-target',
                    false_events=len(times), false_activation=bool(times))
    hits = [t for t in times if end-.1 <= t <= end+.8]
    return dict(hit=bool(hits), first_hit_latency_ms=(hits[0]-end)*1000 if hits else None,
                premature_events=sum(t < end-.1 for t in times), late_events=sum(t > end+.8 for t in times))


def summarize(rows, engine):
    positives = [r[engine]['classification'] for r in rows if 'hit' in r[engine]['classification']]
    latencies = [r['first_hit_latency_ms'] for r in positives if r['hit']]
    result = dict(positive_clips=len(positives), hits=sum(r['hit'] for r in positives),
                  recall=sum(r['hit'] for r in positives)/len(positives),
                  premature_events=sum(r['premature_events'] for r in positives),
                  late_events=sum(r['late_events'] for r in positives),
                  latency_ms_median=float(np.median(latencies)) if latencies else None,
                  latency_ms_p95=float(np.percentile(latencies,95)) if latencies else None)
    for kind in ('common', 'cross-target'):
        negatives = [r[engine]['classification'] for r in rows if r[engine]['classification'].get('negative_kind') == kind]
        result[kind+'_negative'] = dict(clips=len(negatives), false_events=sum(r['false_events'] for r in negatives),
                                       false_activation_clips=sum(r['false_activation'] for r in negatives))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','nyx-model','nyx-calibration','qemu','elf','nm','flash','output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--hiesp-threshold', type=float, default=.63)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    manifest = json.loads(a.manifest.read_text())
    calibration = json.loads(a.nyx_calibration.read_text())
    rows = manifest['records']
    if (not manifest.get('paired_wakephrases') or not manifest.get('endpoint_annotation_source_sha256')
            or manifest.get('independent_final_audio_accessed') or
            any(r['role'] != 'development-paired-validation' or r['split'] != 'validation' for r in rows)):
        raise ValueError('Only endpoint-annotated development paired validation')
    if sha(a.nyx_model) != calibration['model_sha256'] or not calibration.get('validation_only'):
        raise ValueError('Need unchanged Nyx model and validation calibration')
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate records')
    a.output.mkdir(parents=True)
    base = [sys.executable, str(Path(__file__).parents[1]/'qemu_esp32_replay/run_replay.py')]
    for key in ('manifest','qemu','elf','nm','flash'):
        base += ['--'+key,str(getattr(a,key).resolve())]
    base += ['--threshold',str(a.hiesp_threshold)]
    def run(item):
        i,row = item
        with (a.output/f'{i}-runner.log').open('wb') as log:
            proc = subprocess.run(base+['--record',row['path'],'--output',str(a.output/str(i))], stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'Replay failed at {i}; preserve artifacts, no summary')
        return i,json.loads((a.output/str(i)/'report.json').read_text())
    replay = {}
    with ThreadPoolExecutor(4) as pool:
        for f in as_completed([pool.submit(run,item) for item in enumerate(rows)]):
            i,report = f.result(); replay[i] = report
            print(json.dumps(dict(replay_completed=len(replay), planned=len(rows))),flush=True)
    for key in ('elf_sha256','flash_sha256','qemu_sha256','source_sha256','manifest_sha256'):
        if len({r[key] for r in replay.values()}) != 1:
            raise ValueError('Mixed engine/manifest provenance')
    detector = Detector(a.nyx_model)
    results = []
    for i,row in enumerate(rows):
        path = (a.manifest.parent/row['path']).resolve()
        if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed/escaped PCM')
        pcm,rate = sf.read(path,dtype='int16')
        other = replay[i]
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']
                or other['pcm_sha256'] != row['pcm_sha256'] or other['manifest_sha256'] != sha(a.manifest)
                or not other['firmware_crc32_matched'] or other['firmware_stage'] != 104):
            raise ValueError('Invalid same PCM/completion evidence')
        data = features(pcm)
        detector.reset(); raw = detector.predict(data)
        detector.reset(); np.testing.assert_array_equal(raw,detector.predict(data))
        scores = smooth(raw,calibration['sliding_window'])
        trace = a.output/f'{i}-nyx-scores.npy';np.save(trace,scores)
        indices = event_indices(scores,calibration['threshold'])
        nyx_times = [float((3*(j+calibration['sliding_window']-1)+5)*.01) for j in indices]
        esp_times = [r['audio_seconds'] for r in other['events']]
        results.append(dict(record=row, nyx=dict(event_seconds=nyx_times,score_sha256=sha(trace),
                                                reset_repeatable=True,classification=classify(nyx_times,row,'hey_nyx')),
                            hiesp=dict(event_seconds=esp_times,replay_report_sha256=sha(a.output/str(i)/'report.json'),
                                       classification=classify(esp_times,row,'hi_esp'))))
    result = dict(role='Paired synthetic development diagnostic, not acceptance or matched-FAR comparison',
                  source_sha256=sha(Path(__file__)), replay_source_sha256=next(iter(replay.values()))['source_sha256'],
                  frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                  detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  manifest_sha256=sha(a.manifest),nyx_model_sha256=sha(a.nyx_model),
                  nyx_calibration_sha256=sha(a.nyx_calibration),nyx_threshold=calibration['threshold'],
                  hiesp_threshold_requested=a.hiesp_threshold,
                  hiesp_threshold_getter_values=sorted({r['threshold_getter_after_set'] for r in replay.values()}),
                  summaries={engine:summarize(results,engine) for engine in ('nyx','hiesp')},records=results,
                  independent_final_audio_accessed=False,
                  endpoint_protocol=manifest.get('endpoint_protocol', 'Historical endpoint metadata; not phonetic truth'),
                  limits='Reused development TTS presets, no representative recall claim. Native frontends, padding and trigger policies retained, reset per clip. Endpoint protocol disclosed separately; fixed -100..+800ms diagnostic window. Thresholds fixed independently, not matched general false-wake rates. Audio-time latency only; no hardware latency. No ASR-based filtering.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summaries']),flush=True)


if __name__ == '__main__':
    main()
