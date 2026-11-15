"""Prospective selection from a frozen, unconsumed-development metric table.

No inference, audio, model loading or protected-data access. This component
selects a development operating point; it cannot qualify a model for promotion.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

def select(table):
    from scipy.stats import chi2
    if table['role']!='unconsumed-threshold-development':raise ValueError('wrong role')
    points=table['points'];grid=[k/1280 for k in range(1,1281)]
    if [p['threshold'] for p in points]!=grid:raise ValueError('complete ordered prospective grid required')
    hours=float(table['background_hours'])
    if not math.isfinite(hours) or hours<=0:raise ValueError('positive finite exposure required')
    strata=table['expected_positive_strata']
    if not strata or len(strata)!=len(set(strata)):raise ValueError('unique positive strata required')
    denominators=None;negative_count=None;candidates=[]
    def count(x):
        if type(x) is not int or x<0:raise ValueError('nonnegative integer count required')
        return x
    for p in points:
        if set(p['positive'])!=set(strata):raise ValueError('missing or unexpected stratum')
        totals={s:count(p['positive'][s]['clips']) for s in strata}
        if any(v==0 for v in totals.values()):raise ValueError('empty positive stratum')
        if denominators is None:denominators=totals
        if totals!=denominators:raise ValueError('changing positive population')
        hits={s:count(p['positive'][s]['hits']) for s in strata}
        if any(hits[s]>totals[s] for s in strata):raise ValueError('hits exceed clips')
        neg=count(p['confusion_clips']);false=count(p['confusion_activated_clips'])
        if neg==0 or false>neg:raise ValueError('invalid confusion population')
        if negative_count is None:negative_count=neg
        if negative_count!=neg:raise ValueError('changing confusion population')
        events=count(p['background_events']);upper=float(chi2.ppf(.95,2*(events+1))/(2*hours))
        if upper>.5:continue
        worst=min(hits[s]/totals[s] for s in strata)
        total=sum(hits.values())/sum(totals.values())
        # Priority fixed before fresh scores: coverage across voice/carrier
        # strata, overall coverage, confusion, then conservative threshold.
        candidates.append((worst,total,-false,p['threshold'],upper,events))
    if not candidates:return dict(status='no_feasible_background_point',selected_threshold=None,promotion_qualified=False)
    best=max(candidates,key=lambda x:x[:4])
    return dict(status='development_point_selected',selected_threshold=best[3],worst_stratum_recall=best[0],overall_recall=best[1],confusion_activated_clips=-best[2],background_far_upper95=best[4],background_events=best[5],background_hours=hours,promotion_qualified=False,limits='Development selection only; maximum recall may still be poor. Fresh independent final comparison and preregistered acceptance gates remain mandatory. No claim of human generalization or acoustic phonetic timing.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--table',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    data=a.table.read_bytes();table=json.loads(data)
    result=select(table);result.update(table_path=str(a.table),table_sha256=hashlib.sha256(data).hexdigest(),selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
