"""Registered full-phrase development metrics from original raw UINT8 bytes."""
import math
import numpy as np
from scipy.stats import chi2
from uint8_event_policy_v1 import event_steps

GRID=sorted(set([.50,.70,.80,.90,.95,.97,.98,.985,.99,.992,.994]+[.5+i/256 for i in range(128)]))
assert len(GRID)==138


def event_samples(raw,threshold,samples):
    if not isinstance(samples,int) or samples<=0:raise ValueError('Invalid original sample count')
    return [(s,min((3*s+2)*160,samples)) for s in event_steps(raw,threshold)]


def far_upper(events,hours):
    if not isinstance(events,int) or events<0 or not math.isfinite(hours) or hours<=0:
        raise ValueError('Invalid event count/exposure')
    return float(.5*chi2.ppf(.95,2*(events+1))/hours)


def synthetic_metrics(clips,target,threshold,details=False):
    """clips are (authenticated manifest row, original UINT8 vector) pairs."""
    if target not in ('hey_nyx','hey_willow'):raise ValueError('Custom target required')
    positive=[];negative={'common':dict(clips=0,activated_clips=0,events=0),
                         'cross_target':dict(clips=0,activated_clips=0,events=0)}
    early=0;records=[]
    other='hey_willow' if target=='hey_nyx' else 'hey_nyx'
    for row,raw in clips:
        events=event_samples(raw,threshold,row['samples']);times=[v for _,v in events]
        if row['target_id']==target:
            hit=any(t>=row['generated_prefix_end_sample'] for t in times)
            rirhit=any(t>=row['full_rir_prefix_end_sample'] for t in times)
            anyhit=any(t>=row['lead_samples'] for t in times)
            early+=sum(t<row['lead_samples'] for t in times)
            positive.append((row,hit,rirhit,anyhit))
        else:
            kind='cross_target' if row['target_id']==other else 'common'
            negative[kind]['clips']+=1;negative[kind]['activated_clips']+=int(bool(events));negative[kind]['events']+=len(events)
        if details:
            records.append(dict(source_id=row['source_id'],target_id=row['target_id'],variant=row['variant'],carrier=row['carrier'],length_scale=row['length_scale'],events=[dict(raw_step=s,sample=t,**{key+'_signed_ms':(t-row[key])/16 for key in ('generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample')}) for s,t in events]))
    assert len(positive)==36 and negative['common']['clips']==288 and negative['cross_target']['clips']==36
    def rate(group,column=1):
        if not group:raise ValueError('Empty registered metric cell')
        return sum(r[column] for r in group)/len(group)
    conditions={str(v):rate([r for r in positive if r[0]['variant']==v]) for v in range(3)}
    result=dict(positive_clips=36,complete_phrase_activated=sum(r[1] for r in positive),
                macro=float(np.mean(list(conditions.values()))),conditions=conditions,
                standalone=rate([r for r in positive if not r[0]['carrier']]),
                command=rate([r for r in positive if r[0]['carrier']]),
                full_rir_prefix_recall=rate(positive,2),any_after_source_recall=rate(positive,3),
                pre_source_positive_events=early,negatives=negative)
    if details:
        result.update(records=records,speeds={str(s):rate([r for r in positive if r[0]['length_scale']==s]) for s in (.9,1.,1.1)},voices={v:rate([r for r in positive if r[0]['voice']==v]) for v in {r[0]['voice'] for r in positive}})
    return result


def eligible(metrics,upper):
    return (upper<=.5 and metrics['macro']>=.8 and metrics['standalone']>=.75
            and metrics['command']>=.75 and metrics['pre_source_positive_events']==0
            and all(r['events']==0 for r in metrics['negatives'].values()))


def choose(points):
    valid=[p for p in points if p['eligible']]
    return max(valid,key=lambda p:(p['synthetic']['macro'],-p['background_upper95'],p['threshold'])) if valid else None
