"""Implement frozen138-point Joe/new-background gate; null means failure, no fallback."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

ROOT=Path('research/wakeword_p4')
sys.path.insert(0,str(ROOT/'training'))
from evaluate import THRESHOLDS,event_indices
from uint8_event_policy_v1 import EVENT_ARITHMETIC
from score_recovered_new_development_v1 import canonical_means

BASE=Path('.local/wakeword/recovered-new-development-v1')
FREEZE=ROOT/'results/4090-recovered-new-development-execution-pins-v1.json'
GRID=sorted(set(THRESHOLDS+[.5+i/256 for i in range(128)]))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def synthetic_metrics(clips,threshold,details=False):
    positives=[];early=0;negative={'common':dict(clips=0,activated_clips=0,events=0),
                                 'cross_target':dict(clips=0,activated_clips=0,events=0)}
    records=[]
    for row,scores in clips:
        times=[float((3*(i+4)+5)*.01) for i in event_indices(scores,threshold)]
        if row['target']=='hey_nyx':
            hits=[t for t in times if t>=row['lead_samples']/16000]
            positives.append((row,bool(hits)));early+=sum(t<row['lead_samples']/16000 for t in times)
        else:
            kind='common' if row['target']=='confusion' else 'cross_target'
            negative[kind]['clips']+=1;negative[kind]['activated_clips']+=bool(times);negative[kind]['events']+=len(times)
        if details:
            records.append(dict(path=row['path'],target=row['target'],variant=row['variant'],
                text=row['text'],length_scale=row['length_scale'],event_seconds=times,
                signed_generated_end_ms=[(t-row['generated_output_end_sample']/16000)*1000 for t in times],
                signed_full_rir_end_ms=[(t-row['full_rir_end_sample']/16000)*1000 for t in times]))
    def rate(group):return sum(hit for _,hit in group)/len(group)
    assert len(positives)==72 and negative['common']['clips']==216 and negative['cross_target']['clips']==72
    conditions={str(v):rate([(r,h) for r,h in positives if r['variant']==v]) for v in range(3)}
    result=dict(positive_clips=72,activated=sum(h for _,h in positives),macro=float(np.mean(list(conditions.values()))),
                conditions=conditions,pre_source_positive_events=early,negatives=negative,
                standalone=rate([(r,h) for r,h in positives if ', ' not in r['text']]),
                command=rate([(r,h) for r,h in positives if ', ' in r['text']]))
    if details:
        result['records']=records
        result['speeds']={str(s):rate([(r,h) for r,h in positives if r['length_scale']==s]) for s in [.9,1.,1.1]}
        result['texts']={t:rate([(r,h) for r,h in positives if r['text']==t]) for t in sorted({r['text'] for r,_ in positives})}
    return result


def eligible(metrics,upper,minimum):
    return (upper<=.5 and metrics['macro']>=minimum and metrics['standalone']>=.75
            and metrics['command']>=.75 and metrics['pre_source_positive_events']==0
            and all(r['activated_clips']==0 for r in metrics['negatives'].values()))


def choose(points):
    rows=[r for r in points if r['eligible']]
    return max(rows,key=lambda r:(r['synthetic']['macro'],-r['background_upper95'],r['threshold'])) if rows else None


def calibrate():
    assert len(GRID)==138 and GRID[0]==.5 and GRID[-1]==.99609375
    freeze=json.loads(FREEZE.read_text());report=json.loads((BASE/'report.json').read_text())
    assert all(importlib.metadata.version(p)==v for p,v in freeze['versions'].items())
    assert report['execution_pins_sha256']==sha(FREEZE) and report['model_sha256']==freeze['model_sha256']
    assert not report['threshold_selected'] and not report['acceptance_scored']
    assert freeze['event_arithmetic']==report['event_arithmetic']==EVENT_ARITHMETIC
    for p,e in freeze['files'].items():assert sha(p)==e['sha256']
    # Calibration source is pinned in the candidate execution receipt before inference.
    assert freeze['calibrator_sha256']==sha(__file__)
    groups={}
    for phase in ['synthetic','background']:
        manifest=Path(freeze['manifests'][phase]);m=json.loads(manifest.read_text())
        results=report['phases'][phase]['records']
        assert report['phases'][phase]['manifest_sha256']==sha(manifest)
        assert [r['record'] for r in results]==m['records']
        assert len(results)==freeze['expected_records'][phase]
        values=[]
        for i,result in enumerate(results):
            assert result['reset_repeatable']
            p=BASE/phase/f'{i}-scores.npy';assert sha(p)==result['score_sha256']
            scores=np.load(p,allow_pickle=False);row=result['record']
            raw_path=BASE/phase/f'{i}-raw-uint8.npy'
            assert sha(raw_path)==result['raw_uint8_sha256']
            np.testing.assert_array_equal(scores,canonical_means(np.load(raw_path,allow_pickle=False)))
            assert scores.shape==(((row['samples']+159)//160-2)//3-4,) and np.isfinite(scores).all()
            assert np.all((scores>=0)&(scores<=1))
            values.append((row,scores))
        groups[phase]=values
    baseline=json.loads(Path('/tmp/nyx-joe-base-v1/report.json').read_text())
    minimum=max(.8,baseline['summaries']['v30']['macro'])
    hours=sum(row['samples'] for row,_ in groups['background'])/16000/3600
    assert sum(row['samples'] for row,_ in groups['background'])==455204864 and hours>=6
    points=[]
    for threshold in GRID:
        metrics=synthetic_metrics(groups['synthetic'],threshold)
        count=sum(len(event_indices(scores,threshold)) for _,scores in groups['background'])
        upper=float(.5*chi2.ppf(.95,2*(count+1))/hours)
        points.append(dict(threshold=threshold,synthetic=metrics,background_events=count,
                           background_hours=hours,background_upper95=upper,eligible=eligible(metrics,upper,minimum)))
    selected=choose(points)
    result=dict(role='Reserved Joe/new42-recording DEVELOPMENT calibration only',
                model_sha256=report['model_sha256'],execution_pins_sha256=sha(FREEZE),source_sha256=sha(__file__),
                candidate_report_sha256=sha(BASE/'report.json'),grid=points,
                minimum_macro=minimum,threshold=selected['threshold'] if selected else None,
                viable_operating_point=selected is not None,selected=selected,
                validation_only=True,final_data_used=False,sliding_window=5,event_arithmetic=EVENT_ARITHMETIC,
                selected_synthetic_details=synthetic_metrics(groups['synthetic'],selected['threshold'],True) if selected else None,
                acceptance_authorized=False,candidate_promoted=False,
                limits='Development threshold selection only. Null is failure, no fallback, no acceptance rescue. Full generated/RIR reference is not phonetic truth; nominal Poisson FAR ignores clustering; shared-parent/unknown exposure limitations persist.')
    path=BASE/'calibration.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(threshold=result['threshold'],eligible_points=sum(p['eligible'] for p in points),candidate_promoted=False)))


if __name__=='__main__':calibrate()
