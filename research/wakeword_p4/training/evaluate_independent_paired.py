"""Frozen paired TTS intention diagnostic; no phonetic endpoint or latency claims."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from evaluate import Detector, event_indices, smooth
from features import features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows, engine, target):
    positives = [r for r in rows if r['record']['target'] == target]
    result = dict(intended_target_clips=len(positives),
                  target_clips_with_events_after_source_start=sum(bool(r[engine]['post_source_start_events']) for r in positives),
                  target_clips_with_any_event=sum(bool(r[engine]['event_seconds']) for r in positives),
                  target_early_events=sum(r[engine]['early_events'] for r in positives))
    result['intention_clip_activation_rate'] = result['target_clips_with_events_after_source_start']/len(positives)
    for kind in ('common', 'cross-target'):
        negatives = [r for r in rows if (r['record']['target'] is None if kind == 'common'
                                         else r['record']['target'] not in (None,target))]
        result[kind] = dict(clips=len(negatives), events=sum(len(r[engine]['event_seconds']) for r in negatives),
                            activated_clips=sum(bool(r[engine]['event_seconds']) for r in negatives))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'preregistration', 'nyx-model', 'qemu', 'elf', 'nm', 'flash', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required; final artifacts are never overwritten')
    freeze, manifest = json.loads(a.preregistration.read_text()), json.loads(a.manifest.read_text())
    replay_script = Path(__file__).parents[1]/'qemu_esp32_replay/run_replay.py'
    if (sha(a.manifest) != freeze['pcm_manifest_sha256'] or sha(a.nyx_model) != freeze['nyx_model_sha256']
            or sha(Path(__file__)) != freeze['evaluator_source_sha256']
            or sha(replay_script) != freeze['run_replay_source_sha256']
            or sha(Path(__file__).with_name('evaluate.py')) != freeze['detector_source_sha256']
            or sha(Path(__file__).with_name('features.py')) != freeze['frontend_source_sha256']
            or any(sha(getattr(a,k)) != freeze[k+'_sha256'] for k in ('qemu','elf','flash'))):
        raise ValueError('Changed frozen paired model, waveform or scoring policy')
    if manifest['source_holdout_manifest_sha256'] != freeze['source_holdout_manifest_sha256']:
        raise ValueError('Changed frozen original holdout')
    rows = manifest['records']
    if (len(rows) != 468 or len({r['path'] for r in rows}) != 468
            or any(r['split'] != 'testing' or r['role'] != 'independent-paired-final' for r in rows)):
        raise ValueError('Incomplete independent paired test population')
    if (Counter(r['target'] for r in rows) != {'hey_nyx':78, 'hi_esp':78, None:312}
            or len({r['voice'] for r in rows}) != 26
            or Counter(r['speed'] for r in rows) != {.9:156, 1.:156, 1.1:156}
            or any(n != 6 for n in Counter((r['voice'],r['speed']) for r in rows).values())):
        raise ValueError('Changed voice, speed or target population')
    for row in rows:
        path = (a.manifest.parent/row['path']).resolve()
        if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed or escaped paired WAV')
    a.output.mkdir(parents=True)
    base = [sys.executable,str(replay_script),'--manifest',str(a.manifest.resolve()),
            '--final-preregistration',str(a.preregistration.resolve()),
            '--threshold',str(freeze['threshold_requested'])]
    for k in ('qemu','elf','nm','flash'):
        base += ['--'+k,str(getattr(a,k).resolve())]
    def run(item):
        i,row = item
        with (a.output/f'{i}-runner.log').open('wb') as log:
            proc = subprocess.run(base+['--record',row['path'],'--output',str((a.output/str(i)).resolve())],
                                  stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            return i,None
        return i,json.loads((a.output/str(i)/'report.json').read_text())
    replays, failures = {}, []
    with ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(run,item) for item in enumerate(rows)]):
            i,r = future.result()
            if r is None:failures.append(i)
            else:replays[i] = r
            print(json.dumps(dict(completed=len(replays),failures=len(failures),planned=468)),flush=True)
    if failures:
        (a.output/'failures.json').write_text(json.dumps(failures)+'\n')
        raise RuntimeError('Incomplete paired replay; preserve all runs, no quality summary')
    detector, results = Detector(a.nyx_model), []
    for i,row in enumerate(rows):
        r = replays[i]
        pcm,rate = sf.read(a.manifest.parent/row['path'],dtype='int16')
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples'] or r['record'] != row
                or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']
                or r['pcm_sha256'] != row['pcm_sha256'] or r['manifest_sha256'] != sha(a.manifest)
                or not r['firmware_crc32_matched'] or r['firmware_stage'] != 104
                or r['calls'] != (len(pcm)+511)//512 or r['final_chunk_zero_pad_samples'] != (-len(pcm))%512
                or r['threshold_getter_after_set'] != freeze['threshold_getter_expected']
                or r['final_preregistration_sha256'] != sha(a.preregistration)
                or r['source_sha256'] != freeze['run_replay_source_sha256']
                or any(r[k+'_sha256'] != freeze[k+'_sha256'] for k in ('qemu','elf','flash'))):
            raise ValueError('Missing same-PCM official completion evidence')
        data = features(pcm)
        detector.reset()
        values = smooth(detector.predict(data),freeze['sliding_window'])
        detector.reset()
        np.testing.assert_array_equal(values,smooth(detector.predict(data),freeze['sliding_window']))
        trace = a.output/f'{i}-nyx-scores.npy'
        np.save(trace,values)
        indices = event_indices(values,freeze['nyx_threshold'])
        nyx_times = [float((3*(j+freeze['sliding_window']-1)+5)*.01) for j in indices]
        esp_times = [e['audio_seconds'] for e in r['events']]
        def events(times):
            return dict(event_seconds=times,post_source_start_events=[t for t in times if t>=1.8],
                        early_events=sum(t<1.8 for t in times))
        results.append(dict(record=row,nyx=dict(events(nyx_times),score_sha256=sha(trace),reset_repeatable=True),
                            hiesp=dict(events(esp_times),replay_report_sha256=sha(a.output/str(i)/'report.json'))))
    report = dict(role='Independent paired TTS intention activation diagnostic, not phonetic recall/latency acceptance',
                  source_sha256=sha(Path(__file__)),preregistration_sha256=sha(a.preregistration),
                  manifest_sha256=sha(a.manifest),nyx_model_sha256=sha(a.nyx_model),
                  nyx_threshold=freeze['nyx_threshold'],hiesp_threshold_requested=freeze['threshold_requested'],
                  summaries={e:summary(results,e,t) for e,t in [('nyx','hey_nyx'),('hiesp','hi_esp')]},
                  records=results,independent_paired_final_scored=True,thresholds_reselected=False,
                  phonetic_endpoints_available=False,full_goal_achieved=False,
                  limits=freeze['limits'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summaries']),flush=True)


if __name__ == '__main__':
    main()
