"""Reject known consumed model bytes; an unknown model is not automatically qualified."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve original check results')
    registry_path = Path('research/wakeword_p4/results/4090-cross-machine-source-consumption-v1.json')
    registry = json.loads(registry_path.read_text())
    for item in registry['external_evidence'].values():
        assert sha(Path(item['saved_path'])) == item['sha256']
    known = {r['model_sha256']: r for r in registry['matched_models']}
    models = json.loads(args.plan.read_text())['models']
    conflicts = []
    for model in models:
        digest = model.get('expected_sha256', model.get('sha256'))
        assert isinstance(digest, str) and len(digest) == 64
        if digest in known:
            conflicts.append(dict(proposed_voice=model.get('voice', model.get('name')),
                model_sha256=digest, known_consumption=known[digest]))
    report = dict(plan_sha256=sha(args.plan), registry_sha256=sha(registry_path),
        script_sha256=sha(Path(__file__)), checked_models=len(models), conflicts=conflicts,
        known_consumption_check_passed=not bool(conflicts), source_qualified=False,
        roles_reserved=False, synthesis_or_scores_authorized=False,
        limits='Only checks currently registered Mac-consumed model bytes. Absence of a match '
        'is not proof of no training/other-machine exposure, physical identity or pretraining independence.')
    with args.output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(dict(checked_models=len(models), known_consumed_conflicts=len(conflicts),
                         source_qualified=False)))
    raise SystemExit(1 if conflicts else 0)


if __name__ == '__main__': main()
