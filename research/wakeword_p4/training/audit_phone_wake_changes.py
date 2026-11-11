"""Describe matched fixed-reference changes in already-scored development data.

Correlated render views are counted separately from source identities. This
does not change labels, tune a threshold, run inference or qualify a model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_stream_equal_mean import checked, steps
from paired_training import phrase


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'features', 'result', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path')
    plan = json.loads((args.artifacts/'plan.json').read_bytes())
    binding = json.loads(args.result.read_bytes())
    comparison = json.loads(checked(args.artifacts/'comparison.json', binding['raw_comparison_sha256']))
    if sha(args.artifacts/'plan.json') != binding['plan_sha256']:
        raise ValueError('Plan differs from the committed result')
    threshold = plan['control_fallback_reference']
    records = json.loads(checked(args.features/'features.json', plan['feature_index_sha256']['features.json']))
    validation = [r for r in records if r['split'] == 'validation']
    reserved = {r['sha256'] for r in records if r['split'] != 'validation'}
    details = {name: json.loads(checked(args.artifacts/f'validation-{name}'/'validation-details.json',
                   comparison['arms'][name]['details_sha256'])) for name in ('control', 'phonetic')}
    if any(len(d['validation']) != len(validation) for d in details.values()):
        raise ValueError('Different validation clip coverage')
    totals, groups, changed = Counter(), defaultdict(Counter), []
    negative_sources = {'control': set(), 'phonetic': set(), 'new': set(), 'corrected': set()}
    for row, control, treatment in zip(validation, details['control']['validation'],
                                     details['phonetic']['validation'], strict=True):
        if row['sha256'] in reserved:
            raise ValueError('Validation source overlaps another split')
        end = (row['keyword_frame']+2)*.01 if row['label'] else None
        times = (3*(np.arange(row['frames']//3-4)+4)+5)*.01
        for cached in (control, treatment):
            if (any(cached[k] != row[k] for k in ('engine', 'variant', 'label', 'text'))
                    or cached['source'] != row['path'] or cached['keyword_end'] != end
                    or not np.array_equal(cached['times'], times)):
                raise ValueError('Changed clip source or timebase')
            scores = np.asarray(cached['scores'])
            if scores.shape != times.shape or not np.isfinite(scores).all():
                raise ValueError('Invalid cached scores')
            window = scores if end is None else scores[(times >= end-.1) & (times <= end+.8)]
            if (float(window.max()) if len(window) else 0.) != cached['max_score']:
                raise ValueError('Cached maximum does not reproduce')
        hits = [control['max_score'] >= threshold, treatment['max_score'] >= threshold]
        kind = 'positive' if row['label'] else 'negative'
        transition = 'both' if all(hits) else 'gained' if hits[1] else 'lost' if hits[0] else 'neither'
        totals.update({kind+'_views': 1, kind+'_'+transition: 1})
        if not row['label']:
            source = (row['engine'], row['sha256'], phrase(row))
            for name, active in zip(('control', 'phonetic'), hits):
                if active:
                    negative_sources[name].add(source)
            if transition in ('gained', 'lost'):
                negative_sources['new' if transition == 'gained' else 'corrected'].add(source)
            for group in ('phrase:'+phrase(row), 'engine:'+row['engine'], 'variant:'+str(row['variant'])):
                groups[group].update({'views': 1, 'control': hits[0], 'phonetic': hits[1],
                                      'new': transition == 'gained', 'corrected': transition == 'lost'})
        if hits[0] != hits[1]:
            changed.append({k: row[k] for k in ('path', 'sha256', 'engine', 'variant', 'label', 'text', 'speakers')}
                           | {'transition': transition, 'control_maximum': control['max_score'],
                              'phonetic_maximum': treatment['max_score']})
    for name, field in (('control', 'negative_both'), ('phonetic', 'negative_both')):
        extra = totals['negative_lost' if name == 'control' else 'negative_gained']
        if totals[field]+extra != comparison['arms'][name]['reference_false_clips']:
            raise ValueError('False-view transitions differ from the frozen comparison')
        calibration = json.loads(checked(args.artifacts/f'validation-{name}'/'threshold.json',
            comparison['arms'][name]['calibration_sha256']))
        reference = next(r for r in calibration['grid'] if r['threshold'] == threshold)
        hits = totals['positive_both']+totals['positive_lost' if name == 'control' else 'positive_gained']
        if hits/totals['positive_views'] != reference['recall']:
            raise ValueError('Positive-view transitions differ from calibration')
    ambient_index = []
    for name in ('human.json', 'ambient.json'):
        ambient_index += json.loads(checked(args.features/name, plan['feature_index_sha256'][name]))
    ambient = []
    ordinal = 0
    for i, row in enumerate(ambient_index):
        if row['split'] != 'validation':
            continue
        key = f'ambient-validation-{i:03d}.npy'
        feature_path = args.features/row['path']
        feature_sha256 = sha(feature_path)
        if row.get('feature_sha256', feature_sha256) != feature_sha256:
            raise ValueError('Ambient feature bytes differ from the index')
        frames = len(np.load(feature_path, allow_pickle=False, mmap_mode='r'))
        record = {'feature_path': row['path'], 'speaker': row.get('speaker'),
                  'feature_sha256': feature_sha256, 'pcm_sha256': row.get('pcm_sha256'),
                  'source_type': row.get('source_type', 'speech' if 'speaker' in row else 'environment'),
                  'hours': frames*.01/3600, 'arms': {}}
        for name in ('control', 'phonetic'):
            path = args.artifacts/f'validation-{name}'/key
            scores = np.load(path, allow_pickle=False)
            reference = details[name]['ambient'][ordinal]
            if (scores.shape != (frames//3-4,) or not np.isfinite(scores).all()
                    or reference['hours'] != record['hours']):
                raise ValueError('Changed ambient geometry or exposure')
            events = steps(scores, threshold)
            if len(events) != reference['events'][str(threshold)]:
                raise ValueError('Ambient events do not reproduce')
            record['arms'][name] = {'score_sha256': sha(path), 'event_steps': events,
                                   'event_end_seconds': [(3*(step+4)+5)*.01 for step in events]}
            if events and row.get('sources'):
                sources = row['sources']
                separator = round(row['separator_seconds']*16000/len(sources))
                if abs(separator*len(sources)/16000-row['separator_seconds']) > 1e-9:
                    raise ValueError('Cannot reconstruct source separators')
                locations = []
                for step in events:
                    sample = (3*(step+4)+5)*160
                    offset = 0
                    for source in sources:
                        if offset <= sample < offset+source['samples']:
                            locations.append({'utterance': source['utterance'],
                                              'end_seconds_within_utterance': (sample-offset)/16000,
                                              'source_start_sample': offset,
                                              'flac_sha256': source['flac_sha256'],
                                              'transcript_sha256': source['transcript_sha256']})
                            break
                        offset += source['samples']+separator
                    else:
                        locations.append({'in_source_separator_or_padding': True})
                record['arms'][name]['event_end_locations'] = locations
        ambient.append(record)
        ordinal += 1
    if any(len(d['ambient']) != ordinal for d in details.values()):
        raise ValueError('Incomplete ambient coverage')
    report = {'scope': __doc__, 'source_sha256': sha(Path(__file__)),
              'comparison_sha256': sha(args.artifacts/'comparison.json'),
              'result_sha256': sha(args.result),
              'plan_sha256': sha(args.artifacts/'plan.json'), 'threshold': threshold,
              'totals': dict(totals), 'negative_source_counts': {
                  'control': len(negative_sources['control']), 'phonetic': len(negative_sources['phonetic']),
                  'sources_with_gained_false_views': len(negative_sources['new']),
                  'sources_with_lost_false_views': len(negative_sources['corrected']),
                  'added_false_source_identities': len(negative_sources['phonetic']-negative_sources['control']),
                  'removed_false_source_identities': len(negative_sources['control']-negative_sources['phonetic'])},
              'negative_groups': {k: dict(v) for k, v in sorted(groups.items())},
              'changed_views': changed, 'ambient': ambient,
              'limits': 'Reused development scores only, with correlated render variants. Synthetic voice tokens are not human identities. Source counts use engine, original WAV SHA and intended phrase because aliases can share WAVs. Paired source-level improvement/correction sets can overlap when different views change in opposite directions. Cache event time is not a phonetic endpoint. No teacher prediction is used to relabel a source; no model, threshold, runtime, board or audio path is changed.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('totals', 'negative_source_counts')}, indent=2))
    print(json.dumps([r for r in ambient if any(x['event_steps'] for x in r['arms'].values())], indent=2))


if __name__ == '__main__':
    main()
