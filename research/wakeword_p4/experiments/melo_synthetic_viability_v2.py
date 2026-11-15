"""Preregistered point-estimate engineering gates, never human equivalence proof."""
from fractions import Fraction
import math
from scipy.stats import chi2
from select_melo_joint_development_v2 import STRATA

ACCEPTANCE_STRATA={'EN-INDIA/bare':9,'EN-INDIA/command':27,'EN-AU/bare':9,'EN-AU/command':27}

def counts(point,acceptance=False):
 expected=ACCEPTANCE_STRATA if acceptance else STRATA
 if set(point['positive'])!=set(expected):raise ValueError('Wrong prospective strata')
 for s,n in expected.items():
  row=point['positive'][s]
  if row['clips']!=n or type(row['hits']) is not int or not 0<=row['hits']<=n:raise ValueError('Invalid fixed positive population')
 neg=point['confusion_clips'];false=point['confusion_activated_clips'];events=point['background_events']
 if neg!=288 or type(false) is not int or not 0<=false<=neg or type(events) is not int or events<0:raise ValueError('Invalid negative counts')
 positive=point['positive'];command=[v for s,v in positive.items() if s.endswith('/command')]
 return dict(overall=Fraction(sum(v['hits'] for v in positive.values()),72),command=Fraction(sum(v['hits'] for v in command),54),worst=min(Fraction(v['hits'],v['clips']) for v in positive.values()),confusion=Fraction(false,288))

def assess(point,v30,official,hours,acceptance=False):
 if not math.isfinite(hours) or hours<=0:raise ValueError('Positive finite background exposure required')
 a,b,c=counts(point,acceptance),counts(v30,acceptance),counts(official,acceptance)
 upper=float(chi2.ppf(.95,2*(point['background_events']+1))/(2*hours))
 official_upper=float(chi2.ppf(.95,2*(official['background_events']+1))/(2*hours))
 checks=dict(overall_at_least_80pct=a['overall']>=Fraction(4,5),worst_stratum_at_least_75pct=a['worst']>=Fraction(3,4),overall_at_least_v30=a['overall']>=b['overall'],overall_within_5pp_of_official=a['overall']>=c['overall']-Fraction(1,20),command_within_5pp_of_official=a['command']>=c['command']-Fraction(1,20),confusion_within_2point5pp_of_official=a['confusion']<=c['confusion']+Fraction(1,40),candidate_far_upper95_at_most_half=upper<=.5)
 if acceptance:checks.update(acceptance_at_least6hours=hours>=6,official_far_upper95_at_most_half=official_upper<=.5)
 return dict(stage='independent-synthetic-acceptance' if acceptance else 'development-viability',checks=checks,passed=all(checks.values()),candidate_metrics={k:float(v) for k,v in a.items()},candidate_far_upper95=upper,official_far_upper95=official_upper,promotion_qualified=False,limits='Count/FAR gates only. Native/render/ABI/source-role/full-stream execution gates and exact model/threshold lock remain separate requirements. Point-estimate engineering allowances are not statistical equivalence or human phonetic/generalization truth. A failed selected point cannot fall back to runner-up.')
