"""Replay Nyx on frozen development PCM, reusing authenticated official reports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from evaluate import Detector, smooth, event_indices
from features import features
from compare_paired_validation import classify, summarize


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','model','calibration','official-reports','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline',type=Path,help='Optional existing same-model paired result; require exact event reproduction')
    a=p.parse_args()
    if a.output.exists(): p.error('Fresh output required')
    manifest=json.loads(a.manifest.read_text());rows=manifest['records']
    calibration=json.loads(a.calibration.read_text())
    official=json.loads(a.official_reports.read_text())
    expected_counts={'2ff49bdd98b7b8ccaa2d01bef9f8764db331ad2a08c98ace522a05df48763dc8':80,
        '779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be':300}
    count=expected_counts.get(sha(a.manifest))
    if (count is None
            or len(rows)!=count or manifest['independent_final_audio_accessed']
            or len(official)!=count or not calibration['validation_only']
            or calibration['model_sha256']!=sha(a.model) or calibration['threshold'] is None
            or calibration['sliding_window']!=5):
        raise ValueError('Need exact frozen development manifest and calibrated unchanged model')
    for field in ('source_sha256','qemu_sha256','elf_sha256','flash_sha256'):
        if len({r[field] for r in official})!=1: raise ValueError('Mixed official provenance')
    baseline=json.loads(a.baseline.read_text()) if a.baseline else None
    if baseline and (baseline['nyx_model_sha256']!=sha(a.model) or baseline['manifest_sha256']!=sha(a.manifest)
                     or baseline['nyx_threshold']!=calibration['threshold']):
        raise ValueError('Baseline model/protocol differs')
    # Authenticate all sources before scoring; no successful subset.
    for row,other in zip(rows,official):
        path=a.manifest.parent/row['path'];pcm,rate=sf.read(path,dtype='int16')
        if (row!=other['record'] or other['manifest_sha256']!=sha(a.manifest)
                or other['independent_final_audio_accessed'] or other['firmware_stage']!=104
                or not other['firmware_crc32_matched'] or other['model']!='wn9_hiesp'
                or other['threshold_requested']!=.63 or other['threshold_getter_after_set']!=.62496001
                or sha(path)!=row['sha256'] or rate!=16000 or pcm.ndim!=1
                or len(pcm)!=row['samples'] or other['input_samples']!=len(pcm)
                or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=other['pcm_sha256']
                or other['pcm_sha256']!=row['pcm_sha256']):
            raise ValueError('Changed PCM or incompatible cached official report')
    a.output.mkdir(parents=True);detector=Detector(a.model);results=[]
    for i,(row,other) in enumerate(zip(rows,official)):
        pcm,_=sf.read(a.manifest.parent/row['path'],dtype='int16');f=features(pcm)
        detector.reset();raw=detector.predict(f)
        detector.reset();np.testing.assert_array_equal(raw,detector.predict(f))
        scores=smooth(raw,5);indices=event_indices(scores,calibration['threshold'])
        nyx_times=[float((3*(j+4)+5)*.01) for j in indices]
        esp_times=[r['audio_seconds'] for r in other['events']]
        if baseline and (nyx_times!=baseline['records'][i]['nyx']['event_seconds']
                         or esp_times!=baseline['records'][i]['hiesp']['event_seconds']):
            raise ValueError('Cached replay baseline event mismatch')
        trace=a.output/f'{i}-scores.npy';np.save(trace,scores)
        results.append(dict(record=row,nyx=dict(event_seconds=nyx_times,score_sha256=sha(trace),
            classification=classify(nyx_times,row,'hey_nyx')),hiesp=dict(event_seconds=esp_times,
            classification=classify(esp_times,row,'hi_esp'))))
    summary={e:summarize(results,e) for e in ('nyx','hiesp')}
    for e,target in [('nyx','hey_nyx'),('hiesp','hi_esp')]:
        summary[e]['positive_any_event_clips']=sum(bool(r[e]['event_seconds']) for r in results if r['record']['target']==target)
    report=dict(role='Frozen synthetic development diagnostic, historical threshold; not final acceptance',
        model_sha256=sha(a.model),calibration_sha256=sha(a.calibration),threshold=calibration['threshold'],
        manifest_sha256=sha(a.manifest),official_reports_sha256=sha(a.official_reports),
        source_sha256=sha(Path(__file__)),detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
        frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
        baseline_exact_event_reproduction=bool(baseline),reset_exact_clips=count,
        summaries=summary,records=results,final_data_used=False,threshold_selected=False,
        limits='Generation-end window metrics are not phonetic recall/latency. Any-event is an intention diagnostic. Development voices only, independently fixed thresholds, no matched FAR or acceptance claim. Cached official source/model/PCM authenticated; no repeated QEMU run.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
