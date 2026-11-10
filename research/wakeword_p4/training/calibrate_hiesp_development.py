"""Diagnostic validation grid against fixed Hi ESP observed negative event budget.

This separate comparison does not replace the zero-event candidate gate and must
not authorize final testing. Equal counts are an empirical development match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--english-validation',type=Path,required=True)
    p.add_argument('--negative-validation',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    reference=json.loads(a.reference.read_text())
    base=json.loads((a.english_validation/'report.json').read_text())
    threshold_path=a.english_validation/'threshold.json'
    thresholds=json.loads(threshold_path.read_text())['grid']
    if not base.get('validation_only') or reference['independent_final_audio_accessed']:
        raise ValueError('Only existing validation')
    model=base['model_sha256'];window=base['sliding_window']
    expected={}
    for corpus in reference['corpora']:
        records=[r for r in corpus['nyx'] if r['model_sha256']==model]
        if len(records)!=1:raise ValueError('Reference missing model')
        expected[records[0]['report_sha256']]=corpus
    if len(expected)!=len(a.negative_validation):raise ValueError('Need every audited reference corpus')
    sources=[];seen=set();used=set()
    for folder in a.negative_validation:
        path=folder/'report.json';r=json.loads(path.read_text());digest=sha(path)
        if digest not in expected or digest in used:raise ValueError('Changed, repeated or unaudited negative report')
        used.add(digest)
        if (r['model_sha256']!=model or r.get('sliding_window',5)!=window
                or r.get('independent_final_scored') or r.get('development_regression_scored')
                or r.get('testing_or_independent_final_scored') or 'development validation' not in r['role']):
            raise ValueError('Invalid negative evaluation scope/model')
        arrays=[]
        for row in r['speakers']:
            if row['speaker'] in seen:raise ValueError('Speaker/session repeated')
            seen.add(row['speaker']);trace=folder/(row['speaker']+'-scores.npy')
            if sha(trace)!=row['score_sha256']:raise ValueError('Changed score trace')
            values=np.load(trace,allow_pickle=False)
            if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Invalid scores')
            if len(event_indices(values,r['threshold']))!=row['false_events']:raise ValueError('Event recount mismatch')
            arrays.append(values)
        hours=sum(row['source_seconds'] for row in r['speakers'])/3600
        if abs(hours-expected[digest]['hiesp']['source_hours'])>1e-12:raise ValueError('Unequal exposure')
        sources.append(dict(name=folder.name,report_sha256=digest,source_hours=hours,arrays=arrays))
    hours=sum(s['source_hours'] for s in sources)
    if abs(hours-reference['hiesp']['source_hours'])>1e-12:raise ValueError('Incomplete reference exposure')
    budget=reference['hiesp']['events'];grid=[]
    for english in thresholds:
        point=english['threshold'];negative=[]
        for source in sources:
            count=sum(len(event_indices(x,point)) if np.max(x)>=point else 0 for x in source['arrays'])
            negative.append(dict(corpus=source['name'],source_hours=source['source_hours'],events=count))
        count=sum(r['events'] for r in negative)
        base_gate=english['ambient_false_events']==0 and english['max_source_false_rate']<=.01
        grid.append(dict(threshold=point,english=english,negative_corpora=negative,events=count,
                         false_per_hour=count/hours,false_per_hour_upper_95=float(.5*chi2.ppf(.95,2*(count+1))/hours),
                         within_reference_event_budget=count<=budget,
                         exact_reference_event_count=count==budget,
                         diagnostic_eligible=base_gate and count<=budget,
                         passes_existing_zero_event_gate=base_gate and count==0))
    eligible=[r for r in grid if r['diagnostic_eligible']]
    selected=max(eligible,key=lambda r:(r['english']['macro_recall'],-r['english']['max_source_false_rate'],-r['threshold'])) if eligible else None
    result=dict(role='Empirical Hi ESP event-budget development calibration, diagnostic only',
                source_sha256=sha(Path(__file__)),detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                reference_sha256=sha(a.reference),reference_hiesp=reference['hiesp'],
                base_report_sha256=sha(a.english_validation/'report.json'),base_grid_sha256=sha(threshold_path),
                model_sha256=model,sliding_window=window,source_hours=hours,reference_event_budget=budget,
                threshold=selected['threshold'] if selected else None,selected=selected,grid=grid,
                validation_only=True,not_for_final_acceptance=True,existing_zero_event_gate_replaced=False,
                independent_final_audio_accessed=False,
                selection='Within fixed reference observed count budget, retain original ambient0 and source confusing<=1% constraints; maximize original English source-macro recall.',
                limits='Only repeated development negative traces. Empirical count matching does not prove equal population FAR, independent upper-bound acceptance or representative positive recall. Keeps all per-corpus counts. Diagnostic does not override existing zero-event gate or authorize final testing.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('grid','selected')}))
    print(json.dumps(selected))


if __name__=='__main__':main()
