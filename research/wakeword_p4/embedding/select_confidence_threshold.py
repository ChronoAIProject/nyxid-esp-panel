"""Select a confidence-constrained threshold from an immutable validation grid.

No audio, development regression, microphone trials or new model scores are
read. Members, score averaging, event policy and original grid remain frozen.
This operating-point diagnostic does not establish replacement acceptance.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upper95(events, hours):
    if type(events) is not int or events < 0 or not math.isfinite(hours) or hours <= 0:
        raise ValueError('Expected nonnegative event count and positive source exposure')
    from scipy.stats import chi2
    return float(.5 * chi2.ppf(.95, 2 * (events + 1)) / hours)


def select(grid, limit):
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Expected a positive confidence-bound limit')
    eligible = []
    for row in grid:
        bound = upper95(row['continuous_events'], row['continuous_hours'])
        if not math.isclose(bound, row['continuous_events_per_hour_upper_95'], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('Stored validation confidence bound differs from count/exposure')
        if bound <= limit:
            eligible.append(row)
    return max(eligible, key=lambda r: (r['macro_recall'],
        -r['continuous_events_per_hour_upper_95'], -r['max_corpus_false_rate'],
        -r['threshold'])) if eligible else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if digest(Path(__file__)) != plan['script_sha256'] or args.output.exists():
        p.error('Frozen script and fresh output required')
    base = args.data_root.resolve()

    def verified(name, expected):
        path = (base / name).resolve()
        path.relative_to(base)
        if digest(path) != expected:
            raise ValueError('Frozen validation input differs')
        return path

    parent = json.loads(verified(plan['parent_plan'], plan['parent_plan_sha256']).read_text())
    calibration_path = verified(plan['parent_calibration'], plan['parent_calibration_sha256'])
    calibration = json.loads(calibration_path.read_text())
    if (calibration['plan_sha256'] != plan['parent_plan_sha256']
            or calibration['members'] != parent['members']
            or [r['threshold'] for r in calibration['grid']] != parent['event_policy']['threshold_grid']
            or plan['confidence'] != .95):
        raise ValueError('Original members, validation grid or confidence differs')
    for member in calibration['members']:
        verified(member['checkpoint'] + '/head.tflite', member['head_sha256'])
    chosen = select(calibration['grid'], plan['max_continuous_upper95_per_hour'])
    result = copy.deepcopy(calibration)
    result.update(scope=__doc__, plan_sha256=digest(args.plan),
        parent_calibration_sha256=digest(calibration_path), selected=chosen,
        strict_operating_point=bool(chosen and chosen['max_corpus_false_rate'] <= .01),
        accepted=False, selection_rule=plan['selection_rule'])
    report = {'scope': __doc__, 'plan_sha256': digest(args.plan),
        'parent_calibration_sha256': digest(calibration_path),
        'original_selected': calibration['selected'], 'selected': chosen,
        'max_continuous_upper95_per_hour': plan['max_continuous_upper95_per_hour'],
        'members_and_grid_unchanged': True, 'audio_or_test_scores_read': False,
        'accepted': False,
        'limits': ['A validation-selected bound is not an independent test guarantee; threshold-grid selection and repeated experiments add selection effects.',
                   'Poisson bounds assume the evaluated exposure represents use. Read speech/environment coverage remains limited.',
                   'No checkpoint retraining or source change. New operating point still requires frozen development, acoustic, resource and voice-chain tests.']}
    args.output.mkdir(parents=True)
    (args.output / 'calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'original_threshold': calibration['selected']['threshold'],
                     'selected': chosen, 'accepted': False}), flush=True)


if __name__ == '__main__':
    main()
