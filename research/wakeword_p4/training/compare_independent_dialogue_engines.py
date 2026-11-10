"""Audit paired frozen dialogue results without running or selecting models."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from scipy.stats import chi2
import soundfile as sf

from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows, key):
    result = {key(r):r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate meeting result')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','preregistration','nyx-evaluation','hiesp-evaluation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    freeze = json.loads(a.preregistration.read_text())
    manifest = json.loads(a.manifest.read_text())
    npath,hpath = a.nyx_evaluation/'report.json',a.hiesp_evaluation/'report.json'
    nyx,hiesp = json.loads(npath.read_text()),json.loads(hpath.read_text())
    if (sha(Path(__file__)) != freeze['comparison_source_sha256']
            or sha(a.manifest) != freeze['pcm_manifest_sha256']
            or any(r['manifest_sha256'] != sha(a.manifest) for r in (nyx,hiesp))
            or nyx['preregistration_sha256'] != sha(a.preregistration)
            or hiesp['final_preregistration_sha256'] != sha(a.preregistration)
            or nyx['model_sha256'] != freeze['nyx_model_sha256']
            or nyx['threshold'] != freeze['nyx_threshold'] or nyx['sliding_window'] != 5
            or nyx['threshold_reselected'] or not nyx['independent_dialogue_final_scored']
            or not hiesp['independent_final_audio_accessed']
            or nyx['source_sha256'] != freeze['nyx_evaluator_source_sha256']
            or hiesp['source_sha256'] != freeze['run_corpus_source_sha256']
            or sha(Path(__file__).with_name('evaluate.py')) != freeze['detector_source_sha256']):
        raise ValueError('Changed preregistered results or scoring provenance')
    original = keyed(manifest['records'],lambda r:r['session'])
    nrows = keyed(nyx['records'],lambda r:r['session'])
    hrows = keyed(hiesp['reports'],lambda r:r['record']['session'])
    if (original.keys() != nrows.keys() or original.keys() != hrows.keys()
            or len(original) != freeze['expected_streams'] or len(original) != 16):
        raise ValueError('Incomplete same-meeting result population')
    records = []
    for session,row in original.items():
        nr,hr = nrows[session],hrows[session]
        wav = a.manifest.parent/row['path']
        pcm,rate = sf.read(wav,dtype='int16')
        raw = pcm.astype('<i2').tobytes()
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                or sha(wav) != row['sha256'] or hashlib.sha256(raw).hexdigest() != row['pcm_sha256']
                or hashlib.sha256(raw[:row['original_samples']*2]).hexdigest() != row['original_pcm_sha256']
                or np.any(pcm[row['original_samples']:])
                or hr['record'] != row or hr['pcm_sha256'] != row['pcm_sha256']
                or hr['pcm_crc32'] != f'{zlib.crc32(raw):08x}' or not hr['firmware_crc32_matched']
                or hr['firmware_stage'] != 104 or hr['input_samples'] != len(pcm)
                or hr['calls'] != (len(pcm)+511)//512
                or hr['final_chunk_zero_pad_samples'] != (-len(pcm))%512
                or hr['source_sha256'] != freeze['run_replay_source_sha256']
                or hr['final_preregistration_sha256'] != sha(a.preregistration)
                or hr['threshold_requested'] != freeze['threshold_requested']
                or hr['threshold_getter_after_set'] != freeze['threshold_getter_expected']
                or any(hr[k+'_sha256'] != freeze[k+'_sha256'] for k in ('qemu','elf','flash'))):
            raise ValueError('Changed PCM or incomplete official runtime evidence')
        trace = a.nyx_evaluation/(session+'-scores.npy')
        if (sha(trace) != nr['score_sha256'] or nr['pcm_sha256'] != row['pcm_sha256']
                or nr['feature_sha256'] != row['feature_sha256']
                or sha(a.manifest.parent/row['feature_path']) != row['feature_sha256']
                or not nr['prefix_reset_reproducible']):
            raise ValueError('Changed Nyx score/cache provenance')
        scores = np.load(trace,allow_pickle=False)
        if scores.shape != (row['frames']//3-4,) or not np.all(np.isfinite(scores)):
            raise ValueError('Incomplete Nyx trace')
        steps = [int(i) for i in event_indices(scores,freeze['nyx_threshold'])]
        times = [float((3*(i+4)+5)*.01) for i in steps]
        if nr['event_steps'] != steps or nr['event_seconds'] != times or nr['events'] != len(steps):
            raise ValueError('Changed Nyx event counts or timestamps')
        for event in hr['events']:
            if (not 1<=event['call']<=hr['calls'] or event['sample_end'] != event['call']*512
                    or event['audio_seconds'] != event['sample_end']/16000 or event['result'] != 1):
                raise ValueError('Invalid official raw event')
        if nr['source_seconds'] != row['source_seconds'] or hr['source_seconds'] != row['source_seconds']:
            raise ValueError('Unequal source exposure')
        records.append(dict(session=session,meeting_group=session[:-1],participants=row['participants'],
                            source_seconds=row['source_seconds'],pcm_sha256=row['pcm_sha256'],
                            nyx_events=len(steps),hiesp_events=len(hr['events']),
                            nyx_event_seconds=times,hiesp_event_seconds=[e['audio_seconds'] for e in hr['events']]))

    def metrics(rows, engine):
        hours = sum(r['source_seconds'] for r in rows)/3600
        count = sum(r[engine+'_events'] for r in rows)
        upper = float(.5*chi2.ppf(.95,2*(count+1))/hours)
        return dict(source_hours=hours,events=count,false_per_hour=count/hours,
                    false_per_hour_upper_95=upper,passes_numerical_bound=upper<=.5)

    pooled = {e:metrics(records,e) for e in ('nyx','hiesp')}
    for engine,report in [('nyx',nyx['pooled']),('hiesp',hiesp)]:
        total = report['events'] if engine=='nyx' else report['false_events']
        if total != pooled[engine]['events'] or abs(report['source_hours']-pooled[engine]['source_hours'])>1e-12:
            raise ValueError('Raw recount differs from engine summary')
    result = dict(role='Same-PCM independent dialogue comparison; not positive/latency acceptance',
                  source_sha256=sha(Path(__file__)),preregistration_sha256=sha(a.preregistration),
                  manifest_sha256=sha(a.manifest),nyx_report_sha256=sha(npath),hiesp_report_sha256=sha(hpath),
                  pooled=pooled,by_meeting_group={g:{e:metrics([r for r in records if r['meeting_group']==g],e)
                                                  for e in ('nyx','hiesp')}
                                                 for g in sorted({r['meeting_group'] for r in records})},
                  records=records,all16_pcm_crc_completion_cache_and_score_hashes_verified=True,
                  independent_dialogue_final_scored=True,thresholds_reselected=False,
                  full_goal_achieved=False,limits=freeze['limits'])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pooled),flush=True)


if __name__ == '__main__':
    main()
