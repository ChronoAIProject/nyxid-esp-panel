"""Whole-stream development metrics against an exact frozen material manifest.

Keep historical grid, event arithmetic, FAR, eligibility and ranking unchanged.
Only source denominators and standalone carriers come from frozen materials.
"""
import math

import numpy as np

from hey_development_metrics_v1 import GRID, event_samples, far_upper, eligible, choose

BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample',
              'generated_container_end_sample', 'full_rir_container_end_sample')


def validate_clips(clips, expected_rows):
    """Collector/calibrator must authenticate expected_rows before calling."""
    if not expected_rows or [row for row, _ in clips] != expected_rows:
        raise ValueError('Observed source rows differ from complete frozen manifest')
    keys = [(row['source_id'], row.get('variant')) for row in expected_rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate source/variant cell in frozen manifest')
    for row, raw in clips:
        samples = row['samples']
        if type(samples) is not int or samples <= 0:
            raise ValueError('Invalid original sample count')
        frames = max(0, (samples+159)//160-2)
        if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8
                or raw.shape != ((frames+2)//3,)):
            raise ValueError('Expected every complete-tail UINT8 output step')


def synthetic_metrics(clips, target, threshold, expected_rows,
                      standalone_carriers, details=False):
    validate_clips(clips, expected_rows)
    if target not in ('hey_nyx', 'hey_willow'):
        raise ValueError('Custom paired target required')
    if (not isinstance(standalone_carriers, (list, tuple)) or not standalone_carriers
            or len(set(standalone_carriers)) != len(standalone_carriers)):
        raise ValueError('Explicit frozen standalone-carrier values required')
    if not set(standalone_carriers) <= {row['carrier'] for row in expected_rows}:
        raise ValueError('Standalone carrier absent from frozen manifest')
    other = 'hey_willow' if target == 'hey_nyx' else 'hey_nyx'
    variants = sorted({row['variant'] for row in expected_rows})
    negative = {kind: dict(clips=0, activated_clips=0, events=0)
                for kind in ('common', 'cross_target')}
    positive = []; records = []; early = 0
    for row, raw in clips:
        if any(type(row[key]) is not int or not 0 <= row[key] <= row['samples']
               for key in ('lead_samples',)+BOUNDARIES):
            raise ValueError('Invalid frozen sample boundary')
        if not (row['lead_samples'] <= row['generated_prefix_end_sample']
                <= row['generated_container_end_sample'] <= row['full_rir_container_end_sample']
                and row['generated_prefix_end_sample'] <= row['full_rir_prefix_end_sample']
                <= row['full_rir_container_end_sample']):
            raise ValueError('Inconsistent frozen generated boundaries')
        events = event_samples(raw, threshold, row['samples'])
        times = [sample for _, sample in events]
        if row['target_id'] == target:
            hit = any(t >= row['generated_prefix_end_sample'] for t in times)
            rirhit = any(t >= row['full_rir_prefix_end_sample'] for t in times)
            anyhit = any(t >= row['lead_samples'] for t in times)
            early += sum(t < row['lead_samples'] for t in times)
            positive.append((row, hit, rirhit, anyhit))
        else:
            kind = 'cross_target' if row['target_id'] == other else 'common'
            negative[kind]['clips'] += 1
            negative[kind]['activated_clips'] += int(bool(events))
            negative[kind]['events'] += len(events)
        if details:
            records.append(dict(source_id=row['source_id'], target_id=row['target_id'],
                variant=row['variant'], carrier=row['carrier'], length_scale=row['length_scale'],
                events=[dict(raw_step=step, sample=sample,
                    unclamped_support_sample=(3*step+2)*160,
                    **{key+'_signed_ms': (sample-row[key])/16 for key in BOUNDARIES})
                    for step, sample in events]))
    def rate(group, column=1):
        if not group:
            raise ValueError('Empty frozen target metric cell')
        return sum(row[column] for row in group)/len(group)
    conditions = {str(v): rate([r for r in positive if r[0]['variant'] == v]) for v in variants}
    standalone = [r for r in positive if r[0]['carrier'] in standalone_carriers]
    commands = [r for r in positive if r[0]['carrier'] not in standalone_carriers]
    if any(not group['clips'] for group in negative.values()):
        raise ValueError('Frozen paired comparison requires common and cross-target sources')
    result = dict(positive_clips=len(positive), complete_phrase_activated=sum(r[1] for r in positive),
        macro=float(np.mean(list(conditions.values()))), conditions=conditions,
        standalone=rate(standalone), command=rate(commands),
        full_rir_prefix_recall=rate(positive, 2), any_after_source_recall=rate(positive, 3),
        pre_source_positive_events=early, negatives=negative,
        denominators=dict(positive=len(positive), standalone=len(standalone), command=len(commands),
            conditions={str(v): sum(r[0]['variant'] == v for r in positive) for v in variants},
            common=negative['common']['clips'], cross_target=negative['cross_target']['clips']),
        standalone_carriers=list(standalone_carriers))
    if details:
        result.update(records=records,
            speeds={str(s): rate([r for r in positive if r[0]['length_scale'] == s])
                    for s in sorted({r[0]['length_scale'] for r in positive})},
            voices={v: rate([r for r in positive if r[0]['voice'] == v])
                    for v in sorted({r[0]['voice'] for r in positive})})
    return result


def background_metrics(clips, threshold, expected_rows):
    validate_clips(clips, expected_rows)
    samples = sum(row['samples'] for row, _ in clips)
    hours = samples/16000/3600
    count = sum(len(event_samples(raw, threshold, row['samples'])) for row, raw in clips)
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('Invalid complete background exposure')
    return dict(events=count, original_samples=samples, hours=hours,
                upper95=far_upper(count, hours), sources=len(clips))
