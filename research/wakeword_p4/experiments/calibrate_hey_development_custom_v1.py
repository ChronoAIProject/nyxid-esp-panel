"""Registered138-point full-phrase/FAR calibration, null on failure."""
import argparse
import json
from pathlib import Path
import numpy as np
from hey_development_metrics_v1 import GRID,synthetic_metrics,event_samples,far_upper,eligible,choose
from score_hey_development_custom_v1 import FREEZE,OUT,versions,sha


def calibrate(model_id):
    f=json.loads(FREEZE.read_text());assert versions()==f['versions']
    for p,h in f['files'].items():assert sha(p)==h
    base=OUT/model_id;report=json.loads((base/'report.json').read_text())
    assert report['execution_sha256']==sha(FREEZE) and report['model_sha256']==f['models'][model_id]['sha256']
    groups={}
    for phase,key in (('synthetic','synthetic_manifest'),('background','background_manifest')):
        manifest=Path(f[key]);rows=json.loads(manifest.read_text())['records'];observed=report['phases'][phase]
        assert observed['manifest_sha256']==sha(manifest)
        assert [r['record'] for r in observed['records']]==rows
        pairs=[]
        for r in observed['records']:
            assert r['reset_repeat_exact'] and sha(r['trace_path'])==r['trace_sha256']
            raw=np.load(r['trace_path'],allow_pickle=False)
            assert raw.dtype==np.uint8 and raw.shape==(r['raw_steps'],)
            assert len(raw)==((r['record']['samples']+159)//160-2)//3
            pairs.append((r['record'],raw))
        groups[phase]=pairs
    target='hey_willow' if model_id=='hey_willow' else 'hey_nyx'
    hours=sum(r['samples'] for r,_ in groups['background'])/16000/3600;assert hours>=8
    points=[]
    thresholds=sorted(set(GRID+([.91796875] if model_id=='v30' else [])))
    for threshold in thresholds:
        m=synthetic_metrics(groups['synthetic'],target,threshold)
        count=sum(len(event_samples(raw,threshold,r['samples'])) for r,raw in groups['background'])
        upper=far_upper(count,hours)
        points.append(dict(threshold=threshold,synthetic=m,background_events=count,background_hours=hours,background_upper95=upper,eligible=eligible(m,upper),registered_grid=threshold in GRID))
    selected=choose([p for p in points if p['registered_grid']])
    result=dict(schema='hey-development-custom-calibration-v1',model_id=model_id,model_sha256=report['model_sha256'],execution_sha256=sha(FREEZE),report_sha256=sha(base/'report.json'),grid=points,selected=selected,threshold=selected['threshold'] if selected else None,eligible_points=sum(p['eligible'] for p in points if p['registered_grid']),selected_details=synthetic_metrics(groups['synthetic'],target,selected['threshold'],True) if selected else None,v30_original_fixed_point=next((p for p in points if model_id=='v30' and p['threshold']==.91796875),None),candidate_promoted=False,acceptance_authorized=False,limits='Development only; full-phrase means internal generated boundary, not acoustic truth. Nominal Poisson FAR does not model clustering. Null failure cannot open acceptance or rescue via another checkpoint.')
    with (base/'calibration.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(model=model_id,threshold=result['threshold'],eligible_points=result['eligible_points'],candidate_promoted=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('hey_willow','hey_nyx','v30'),required=True);calibrate(p.parse_args().model)
