"""Constructed scoring regression; reads pinned policy source, never audio.

The 396-cell fixture and independent event oracle cover new denominators,
explicit bare carriers, complete final steps and unchanged legacy decisions.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def expect(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected invalid input to be rejected')


def oracle(raw, threshold, samples):
    """Direct window slices; no use of the implementation's running sum."""
    result = []
    armed = True
    last = -1000
    for end in range(5, len(raw)+1):
        score = sum(int(v) for v in raw[end-5:end])/1280
        if end-last >= 67:
            if score < threshold/2:
                armed = True
            if armed and score >= threshold:
                result.append((end, min((3*end+2)*160, samples)))
                last, armed = end, False
    return result


def fixture():
    rows = []
    targets = ['hey_willow', 'hey_nyx', 'hi_esp', 'hey', 'hello', 'hey_william',
               'hey_pillow', 'hey_nick', 'hey_next', 'hey_mix', 'hey_muse']
    for target in targets:
        for carrier in ('bare', 'light', 'time', 'timer'):
            for scale in (.9, 1., 1.1):
                for variant in range(3):
                    rows.append(dict(source_id=f'{target}/{carrier}/{scale}', target_id=target,
                        carrier=carrier, length_scale=scale, variant=variant, voice='constructed',
                        samples=50000, lead_samples=16000, generated_prefix_end_sample=32000,
                        full_rir_prefix_end_sample=33232, generated_container_end_sample=34000,
                        full_rir_container_end_sample=35232))
    # Independently enumerated: 311 real frames, ceil(311/3)=104 calls.
    return rows, [(row, np.zeros(104, dtype=np.uint8)) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ref', default='3eff762')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    here = Path(__file__).resolve().parent
    source = (here/'complete_development_metrics_v1.py').read_bytes()
    sources = {name: subprocess.check_output(['git', 'show', commit+
        ':research/wakeword_p4/experiments/'+name]) for name in
        ('uint8_event_policy_v1.py', 'hey_development_metrics_v1.py')}
    checks = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, data in sources.items():
            (root/name).write_bytes(data)
        (root/'complete_development_metrics_v1.py').write_bytes(source)
        load('uint8_event_policy_v1', root/'uint8_event_policy_v1.py')
        legacy = load('hey_development_metrics_v1', root/'hey_development_metrics_v1.py')
        metrics = load('constructed_complete_metrics', root/'complete_development_metrics_v1.py')
        rows, clips = fixture()
        assert len(rows) == 396
        for target in ('hey_willow', 'hey_nyx'):
            traces = [(row, raw.copy()) for row, raw in clips]
            for row, raw in traces:
                if row['target_id'] == target:
                    raw[75:80] = 255
            result = metrics.synthetic_metrics(traces, target, .99, rows, ['bare'], True)
            assert result['denominators'] == dict(positive=36, standalone=9, command=27,
                conditions={'0':12, '1':12, '2':12}, common=324, cross_target=36)
            assert result['macro'] == result['standalone'] == result['command'] == 1
            assert result['voices'] == {'constructed':1.0}
            assert metrics.eligible(result, .3)
            muse_index = next(i for i, (row, _) in enumerate(traces) if row['target_id'] == 'hey_muse')
            traces[muse_index][1][75:80] = 255
            rejected = metrics.synthetic_metrics(traces, target, .99, rows, ['bare'])
            assert rejected['negatives']['common'] == dict(clips=324, activated_clips=1, events=1)
            assert not metrics.eligible(rejected, .3)
            traces[muse_index][1][:] = 0
            for row, raw in traces:
                if row['target_id'] == target:
                    raw[:] = 0
                    raw[-5:] = 255
            tail = metrics.synthetic_metrics(traces, target, .99, rows, ['bare'], True)
            assert tail['macro'] == 1
            tail_event = next(r['events'][0] for r in tail['records'] if r['target_id'] == target)
            assert tail_event['raw_step'] == 104 and tail_event['sample'] == 50000
            assert tail_event['unclamped_support_sample'] == 50240
            assert tail_event['generated_prefix_end_sample_signed_ms'] == 1125
            assert not metrics.event_samples(traces[next(i for i,(r,_) in enumerate(traces)
                if r['target_id'] == target)][1][:-1], .99, 50000)
            for row, raw in traces:
                if row['target_id'] == target:
                    raw[:] = 0
                    raw[0:5] = 255
                    raw[75:80] = 255
            early = metrics.synthetic_metrics(traces, target, .99, rows, ['bare'])
            assert early['macro'] == 1 and early['pre_source_positive_events'] == 36
            assert not metrics.eligible(early, .3)
        checks.append('both 396-cell targets:36/9/27/324/36, bare carriers, added Muse rejection, final-only step104, early-positive rejection')

        split = [(r, raw.copy()) for r, raw in clips]
        for row, raw in split:
            if row['target_id'] == 'hey_nyx':
                raw[63:68] = 255
        result = metrics.synthetic_metrics(split, 'hey_nyx', .99, rows, ['bare'], True)
        assert result['macro'] == 1 and result['full_rir_prefix_recall'] == 0
        assert result['records'][36]['events'][0]['full_rir_prefix_end_sample_signed_ms'] == -17
        checks.append('dry-prefix and full-RIR-prefix recall and signed offsets remain distinct')

        invalid = []
        def reject(name, changed_clips=clips, expected=rows, carriers=('bare',)):
            expect(lambda: metrics.synthetic_metrics(changed_clips, 'hey_nyx', .99, expected, carriers))
            invalid.append(name)
        reject('dropped row', clips[:-1])
        reject('reordered rows', list(reversed(clips)))
        changed = copy.deepcopy(clips); changed[0][0]['voice'] = 'different'
        reject('changed original row', changed)
        duplicate_rows = rows+[rows[0]]
        reject('duplicate source/variant', clips+[clips[0]], duplicate_rows)
        reject('wrong raw dtype', [(rows[0], clips[0][1].astype(float))]+clips[1:])
        reject('missing complete-tail step', [(rows[0], clips[0][1][:-1])]+clips[1:])
        reject('extra step', [(rows[0], np.zeros(105, dtype=np.uint8))]+clips[1:])
        reject('unknown standalone', carriers=('bare', 'typo'))
        reject('no standalone', carriers=())
        for key, value in [('lead_samples', -1), ('generated_prefix_end_sample', 34001),
                           ('full_rir_prefix_end_sample', 50001), ('generated_container_end_sample', 34000.0)]:
            changed = copy.deepcopy(clips); changed[0][0][key] = value
            reject(key+' invalid', changed, [r for r,_ in changed])
        checks.append('13 malformed/incomplete/changed manifest, carrier or output cases rejected')
        assert len(invalid) == 13

        rng = np.random.default_rng(20261114)
        old = [(dict(row, carrier='' if row['carrier']=='bare' else row['carrier']),
                rng.integers(0, 256, 104, dtype=np.uint8))
               for row,_ in clips if row['target_id'] != 'hey_muse']
        assert len(old) == 360 and len(metrics.GRID) == 138 and metrics.GRID == legacy.GRID
        for target in ('hey_willow', 'hey_nyx'):
            for threshold in metrics.GRID:
                previous = legacy.synthetic_metrics(old, target, threshold)
                current = metrics.synthetic_metrics(old, target, threshold, [r for r,_ in old], [''])
                assert all(current[k] == v for k,v in previous.items())
                assert metrics.eligible(current, .3) == legacy.eligible(previous, .3)
            previous = legacy.synthetic_metrics(old, target, .5, True)
            current = metrics.synthetic_metrics(old, target, .5, [r for r,_ in old], [''], True)
            for record in current['records']:
                for event in record['events']:
                    event.pop('unclamped_support_sample')
            assert all(current[k] == v for k,v in previous.items())
        for threshold in metrics.GRID:
            for raw in (old[0][1], old[1][1], np.r_[np.full(5,255),np.zeros(70),np.full(5,255)].astype(np.uint8)):
                assert metrics.event_samples(raw, threshold, 50000) == oracle(raw, threshold, 50000)
        checks.append('all138 grid points x both targets match unchanged360 legacy fields/eligibility; detailed records and independent event oracle match')

        background_rows = [dict(source_id='bg0', samples=50000), dict(source_id='bg1', samples=50000)]
        bg = [(r, np.zeros(104, dtype=np.uint8)) for r in background_rows]
        bg[0][1][-5:] = 255
        result = metrics.background_metrics(bg, .99, background_rows)
        assert result['events'] == 1 and result['sources'] == 2
        assert result['original_samples'] == 100000 and result['hours'] == 100000/16000/3600
        assert result['upper95'] == legacy.far_upper(1, result['hours'])
        assert abs(metrics.far_upper(0, 10)-.299573227355399) < 1e-15
        points = [dict(eligible=True, synthetic={'macro':r}, background_upper95=u, threshold=t)
                  for r,u,t in [(1,.3,.5),(1,.2,.6),(1,.2,.7),(.9,.1,.9)]]
        points.append(dict(eligible=False, synthetic={'macro':1}, background_upper95=0, threshold=1))
        assert metrics.choose(points) == legacy.choose(points) == points[2]
        assert metrics.choose([]) is None and metrics.choose([points[-1]]) is None
        checks.append('whole original-sample background exposure including final event, exact Poisson upper bound, unchanged ranking and null selector')

    report = dict(schema='complete-development-metrics-constructed-review-v1',
        gpu_source_commit=commit, metrics_sha256=hashlib.sha256(source).hexdigest(),
        upstream_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=checks, invalid_rejections=invalid, all_checks_pass=True,
        development_or_acceptance_audio_accessed=False, learned_model_executed=False,
        limits='Constructed396/360-cell rows and UINT8 traces. Pinned unchanged policy/metrics sources, no model/audio/frontend/P4 inference or recognition/FAR acceptance. Expected manifest authentication belongs to caller.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks), all_checks_pass=True)))


if __name__ == '__main__':
    main()
