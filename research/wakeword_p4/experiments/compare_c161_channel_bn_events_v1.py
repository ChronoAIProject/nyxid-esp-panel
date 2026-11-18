"""Exact-clock offline simulation of the existing C161-first arbiter for two frozen parents."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from replay_c161_binary40_parallel_events_v2 import COHORTS, merge, read, sha

R = Path('research/wakeword_p4/results')
PREFIX = '4090-broadcast-channel-bn-'
STEM = '4090-c161-channel-bn-events'
ORIGINAL = R/'4090-c161-binary40-parallel-events-original-v2.json'
CONTEXT = R/'4090-parallel-context-streams-original-v1.json'
PROTOCOL = Path(__file__).with_name('c161_channel_bn_event_protocol_v1.md')
MODEL_SHA = 'dc4e5874141828ae6bf88d03011411dbb4a617c262215467fae09fec3526945c'
BASE_SHA = 'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'


def trace(path, expected, clock_name, offset, code_name='uint8'):
    assert sha(path) == expected, path
    with np.load(path, allow_pickle=False) as arr:
        values = arr[code_name]
        clock = arr[clock_name]-offset
    assert values.dtype == np.uint8 and values.ndim == clock.ndim == 1 and len(values) == len(clock)
    # Integer cumulative sums keep the common arm, rearm and refractory arithmetic exact.
    totals = np.r_[0, np.cumsum(values, dtype=np.int64)]
    armed, last, events = True, -1000, []
    for step, total in enumerate(totals[5:]-totals[:-5], 5):
        if step-last >= 67:
            if 2*total < 1175:
                armed = True
            if armed and total >= 1175:
                events.append(dict(step=step, relative_sample=int(clock[step-1])))
                last, armed = step, False
    return clock, events


def clip_group(cohort, row):
    s = row['source']
    if row['phase'] == 'background':
        return 'background'
    if cohort in ['builtin', 'daily']:
        return s['group']
    if cohort == 'mac':
        return s['voice']+'/'+s['target']
    if cohort == 'melo':
        return s['voice_id'].split(':')[-1]+'/'+('target' if s['phrase_id']=='nyx' else s['phrase_id'])+'/'+s['carrier']
    return ('target' if s['target_id']=='hey_nyx' else s['target_id'])+'/'+s.get('carrier','unclassified')


def owned(events, clip):
    return any(clip['start'] < e['relative_sample'] <= clip['end'] for e in events)


def aggregate(groups, key, hits, events=None, hours=None):
    g = groups.setdefault(key, dict(clips=0, hits={n:0 for n in hits}, gained=0, lost=0))
    g['clips'] += 1
    for n,value in hits.items():
        g['hits'][n] += value
    g['gained'] += hits['pair'] and not hits['c161']
    g['lost'] += hits['c161'] and not hits['pair']
    if hours is not None:
        g['hours'] = g.get('hours', 0) + hours
        counts = g.setdefault('events', {n:0 for n in hits})
        for n,ev in events.items():
            counts[n] += sum(e['relative_sample']>0 for e in ev)


def merge_stream(first, second):
    emitted, suppressed = merge({'c161':first, 'channel_bn':second})
    # Independent step-bucket traversal, C161 wins simultaneous returns.
    by_step = {e['step']:e for e in second}
    by_step.update({e['step']:e for e in first})
    last = -1000
    selected = []
    for step in sorted(by_step):
        if step-last >= 67:
            selected.append((step, by_step[step]['relative_sample']))
            last = step
    assert selected == [(e['step'],e['relative_sample']) for e in emitted]
    return dict(c161=first, channel_bn=second, pair=emitted), suppressed


def main():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PROTOCOL)]) == PROTOCOL.read_bytes()
    review_path = R/(PREFIX+'consumed-review-v1.json')
    review = read(review_path)
    assert review['all_passed'] and review['execution_stages_before_review'] == 17
    assert sha('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite') == BASE_SHA
    export = read(R/(PREFIX+'joint-export-original-v1.json'))['model']
    assert sha(export['path']) == export['sha256'] == MODEL_SHA
    baseline = read(ORIGINAL)
    groups, records, inputs = {}, [], {str(p):sha(p) for p in [PROTOCOL, ORIGINAL, CONTEXT, review_path, Path(__file__)]}
    for cohort in COHORTS:
        comparison = R/(PREFIX+f'joint-{cohort}-comparison-v1.json')
        inputs[str(comparison)] = sha(comparison)
        assert inputs[str(comparison)] == review['arms']['joint'][cohort]['comparison_sha256']
        current = read(comparison)
        members = current['archive']['members']
        old_root = Path(f'.local/wakeword/binary40-{cohort}-replay-v1')
        new_root = Path(f'.local/wakeword/broadcast-channel-bn-joint-{cohort}-replay-v1')
        for old, new in zip(baseline['cohorts'][cohort]['records'], current['records'], strict=True):
            base, prev = old['paths']['c161'], old['paths']['binary40']
            index = old['index']
            if cohort == 'continuous':
                assert index == new['gap_samples'] and old['source']['pcm_sha256']==new['pcm_sha256']
            else:
                assert index == new['index'] and old['source']==new.get('source',new.get('request'))
            rel = Path(prev['path']).relative_to(old_root)
            refclock, first = trace(base['path'], base['sha256'], base['clock'], base['offset'])
            clock, second = trace(new_root/rel, members[str(rel)]['sha256'], prev['clock'], prev['offset'])
            assert np.array_equal(refclock, clock) and first == old['saved']['c161']
            if cohort in ['primed','jenny','joe','melo']:
                assert second == new['events']['binary40']
            events, suppressed = merge_stream(first, second)
            record = dict(cohort=cohort,index=index,phase=old['phase'],source=old['source'],events=events,
                          suppressed=suppressed,c161_trace=base['sha256'],bn_trace=members[str(rel)]['sha256'])
            if cohort == 'continuous':
                timeline = old['timeline']
                record['clips'] = []
                for clip in (c for c in timeline if c['kind']=='clip'):
                    hits = {n:owned(ev,clip) for n,ev in events.items()}
                    aggregate(groups, f'{cohort}/{index}/'+clip['group'],hits)
                    record['clips'].append(dict(clip=clip,hits=hits))
                record['outside'] = {n:[e for e in ev if not any(c['kind']=='clip' and c['start']<e['relative_sample']<=c['end'] for c in timeline)] for n,ev in events.items()}
            else:
                s = old['source']
                lead = s.get('lead_samples',0) if cohort in ('primed','jenny','joe') and s.get('target_id')=='hey_nyx' else 0
                hits = {n:any(e['relative_sample']>0 and e['relative_sample']>=lead for e in ev) for n,ev in events.items()}
                assert hits['c161']==old['hits']['c161']
                if new.get('hits') is not None:
                    assert hits['channel_bn']==new['hits']['binary40']
                record['hits'] = hits
                hours = s['samples']/16000/3600 if old['phase']=='background' else None
                aggregate(groups,cohort+'/'+clip_group(cohort,old),hits,events,hours)
            records.append(record)
    assert len(records)==baseline['total_rows']==1760
    ctx_path = R/(PREFIX+'joint-context-original-v1.json')
    inputs[str(ctx_path)] = sha(ctx_path)
    for old, new in zip(read(CONTEXT)['streams'],read(ctx_path)['streams'],strict=True):
        assert old['name']==new['name'] and old['timeline']==new['timeline']
        a,b=old['trace'],new['trace']
        refclock,first=trace(a['path'],a['sha256'],'relative_block_return',0,'c161_uint8')
        clock,second=trace(b['path'],b['sha256'],'relative_block_return',0)
        assert np.array_equal(refclock,clock) and first==old['events']['c161'] and second==new['events']
        events,suppressed=merge_stream(first,second)
        clips=[]
        for clip in (c for c in old['timeline'] if c['kind']=='clip'):
            hits={n:owned(ev,clip) for n,ev in events.items()}
            aggregate(groups,'context/'+old['name']+'/'+clip['group'],hits)
            clips.append(dict(clip=clip,hits=hits))
        assert len(clips)==48
        outside={n:[e for e in ev if not any(c['clip']['start']<e['relative_sample']<=c['clip']['end'] for c in clips)] for n,ev in events.items()}
        records.append(dict(cohort='context',name=old['name'],events=events,suppressed=suppressed,
                            clips=clips,outside=outside,c161_trace=a['sha256'],bn_trace=b['sha256']))
    packed = R/(STEM+'-streams-v1.jsonl.gz')
    with packed.open('xb') as target, gzip.GzipFile(fileobj=target,mode='wb',filename='',mtime=0) as z:
        for row in records:
            z.write((json.dumps(row, separators=(',',':'))+'\n').encode())
    result=dict(inputs=inputs,c161_sha256=BASE_SHA,channel_bn_sha256=MODEL_SHA,streams=len(records),
                groups=groups,stream_records_sha256=sha(packed),new_model_invocations=0,
                scope='Saved complete consumed traces, original event policy, independent step-bucket merge. No new exposure, threshold adjustment or hardware claim.')
    with (R/(STEM+'-original-v1.json')).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':
    main()
