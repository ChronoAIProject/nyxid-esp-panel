"""Exercise the frozen future gate with consumed v30 traces and analytic fixtures.

No candidate scoring, threshold search, inference, audio I/O or board access.
Run from the repository root with --joe-replay <authenticated replay directory>
and --output <fresh directory>. Only the original v30 fixed point is replayed.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


UPSTREAM = 'f3f7241'
CALIBRATOR = 'research/wakeword_p4/experiments/calibrate_recovered_new_development_v1.py'
EVALUATOR = 'research/wakeword_p4/training/evaluate.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'])


def functions(blob, names, namespace):
    nodes = [node for node in ast.parse(blob).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen pure functions>', 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joe-replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    calibrator, evaluator = source(UPSTREAM, CALIBRATOR), source(UPSTREAM, EVALUATOR)
    assert sha(calibrator) == '92cf6f4517a5828741a26f027120502407ba557bff1262a108c5f26b0f0083da'
    assert sha(evaluator) == 'a7591fcadf00c1393887f9df99f201dced6d42a2111032f3803d842dd0086256'
    namespace = {'np': np}
    functions(evaluator, ['event_indices'], namespace)
    functions(calibrator, ['synthetic_metrics', 'eligible', 'choose'], namespace)
    metrics, eligible, choose = (namespace[key] for key in ('synthetic_metrics', 'eligible', 'choose'))
    manifest = json.loads(source('90bca76', 'research/wakeword_p4/results/4090-joe-fixed-baseline-evidence-manifest-v1.json'))
    pins = {row['path']: row for row in manifest['files']}

    def authenticated(name):
        data = (args.joe_replay / name).read_bytes()
        pin = pins['replay/' + name]
        assert len(data) == pin['size'] and sha(data) == pin['sha256']
        return data

    baseline = json.loads(authenticated('report.json'))
    clips = []
    for index, row in enumerate(baseline['records']):
        name = f'{index}-v30-scores.npy'
        assert sha(authenticated(name)) == row['v30']['score_sha256']
        scores = np.load(args.joe_replay / name, allow_pickle=False)
        assert scores.shape == (((row['record']['samples'] + 159) // 160 - 2) // 3 - 4,)
        assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
        clips.append((row['record'], scores))
    assert len(clips) == 360
    result = metrics(clips, .91796875, details=True)
    old = baseline['summaries']['v30']
    assert (result['activated'], result['macro'], result['standalone'], result['command']) == (
        old['post_source_activated'], old['macro'], old['standalone_rate'], old['command_rate'])
    assert result['pre_source_positive_events'] == old['pre_source_positive_events']
    for kind in ['common', 'cross_target']:
        assert result['negatives'][kind] == old[kind]
    assert result['conditions'] == {key: value['activation_rate'] for key, value in old['conditions'].items()}
    for current, previous in zip(result['records'], baseline['records'], strict=True):
        for new, original in [('event_seconds', 'event_seconds'),
                              ('signed_generated_end_ms', 'all_event_signed_generated_end_ms'),
                              ('signed_full_rir_end_ms', 'all_event_signed_full_rir_end_ms')]:
            np.testing.assert_allclose(current[new], previous['v30'][original], rtol=0, atol=1e-10)

    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    # All presentations below are analytic score arrays, not model outputs.
    # Set source start at 0.5s, exactly the event clock at score index 11.
    def fixture(positive_steps, negative_kind=None):
        rows = []
        negative_added = False
        for original, _ in clips:
            row = {**original, 'lead_samples': 8000}
            scores = np.zeros(150)
            if row['target'] == 'hey_nyx':
                scores[positive_steps] = 1
            elif negative_kind and not negative_added and (
                    (row['target'] == 'confusion') == (negative_kind == 'common')):
                scores[[11, 79]] = 1
                negative_added = True
            rows.append((row, scores))
        assert not negative_kind or negative_added
        return rows

    good = metrics(fixture([11]), .8, details=True)
    check('source-boundary events count as hits', good['activated'] == 72 and good['macro'] == 1
          and good['standalone'] == 1 and good['command'] == 1 and good['pre_source_positive_events'] == 0)
    check('all three conditions, three speeds and four texts retained', len(good['conditions']) == 3
          and len(good['speeds']) == 3 and len(good['texts']) == 4 and len(good['records']) == 360)
    check('clean analytic presentations eligible', eligible(good, .4, .8))
    early = metrics(fixture([10]), .8)
    check('pre-source events do not count as hits', early['activated'] == 0
          and early['pre_source_positive_events'] == 72 and not eligible(early, .4, .8))
    repeated = metrics(fixture([0, 68]), .8, details=True)
    check('later hit cannot hide an early event', repeated['activated'] == 72
          and repeated['pre_source_positive_events'] == 72 and not eligible(repeated, .4, .8))
    check('all repeated events and signed latencies retained', all(
        len(row['event_seconds']) == len(row['signed_generated_end_ms']) == len(row['signed_full_rir_end_ms']) == 2
        for row in repeated['records'] if row['target'] == 'hey_nyx'))
    for kind in ['common', 'cross_target']:
        negative = metrics(fixture([11], kind), .8)
        check(kind + ' one activated clip with two events rejects',
              negative['negatives'][kind] == dict(clips=216 if kind == 'common' else 72,
                  activated_clips=1, events=2) and not eligible(negative, .4, .8))
    edge = copy.deepcopy(good)
    edge.update(macro=.8, standalone=.75, command=.75)
    check('all equality boundaries included', eligible(edge, .5, .8))
    for key in ['macro', 'standalone', 'command']:
        bad = copy.deepcopy(edge)
        bad[key] = math.nextafter(bad[key], -math.inf)
        check(key + ' one binary64 step below minimum rejects', not eligible(bad, .5, .8))
    check('FAR one binary64 step above maximum rejects', not eligible(edge, math.nextafter(.5, math.inf), .8))
    check('higher frozen-baseline macro remains binding', not eligible(edge, .4, .9))
    points = [dict(threshold=t, eligible=True, synthetic=dict(macro=m), background_upper95=f)
              for t, m, f in [(.9, .9, .3), (.8, .95, .4), (.7, .95, .2), (.6, .95, .2)]]
    check('selection ranks macro then FAR then higher threshold', choose(points)['threshold'] == .7)
    check('ineligible higher-recall point cannot win', choose(points + [dict(threshold=.99,
          eligible=False, synthetic=dict(macro=1), background_upper95=0)])['threshold'] == .7)
    check('no eligible point returns null', choose([{**row, 'eligible': False} for row in points]) is None)
    check('empty grid returns null', choose([]) is None)
    args.output.mkdir(parents=True)
    report = dict(schema='mac-future-development-gate-audit-v1', upstream_commit=UPSTREAM,
        calibrator_sha256=sha(calibrator), evaluator_sha256=sha(evaluator),
        auditor_sha256=sha(Path(__file__).read_bytes()), numpy=np.__version__,
        authenticated_existing_files=361, fixed_v30_threshold=.91796875,
        existing_360_clip_metrics_and_event_details_match=True,
        existing_v30=dict(positive_activated=result['activated'], common=result['negatives']['common'],
                          cross_target=result['negatives']['cross_target']),
        analytic_checks=checks, analytic_checks_passed=len(checks),
        candidate_scored=False, threshold_search=False, final_data_accessed=False,
        device_operated=False, audio_io_performed=False, accepted=False,
        limits=['Only the frozen pure metric/eligibility/selection functions are executed, not full calibration orchestration, inference or source gates.',
                'The existing 360-clip replay uses only frozen v30 threshold .91796875. Other arrays/metrics are explicit analytic fixtures.',
                'No GPU scorer/calibrator, live training source or original report is changed; metadata/numeric proposals still require GPU agreement.',
                'This prevents gate regressions; it is not evidence of a better model or Hi ESP parity.'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
