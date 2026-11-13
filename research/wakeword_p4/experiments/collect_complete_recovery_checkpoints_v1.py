"""Preserve complete original checkpoint lines; never select, restart or train."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

RESULTS = Path('research/wakeword_p4/results')


def collect(condition):
    source = Path(f'.local/wakeword/model-complete-recovery-{condition}-formal-v1/progress.jsonl')
    if not source.exists():
        return 0
    last = 0
    for raw in source.read_bytes().splitlines(keepends=True):
        if not raw.endswith(b'\n'):
            continue
        record = json.loads(raw)
        step = record['step']
        assert isinstance(step, int) and step > last and step % 5000 == 0 and step <= 120000
        last = step
        tag = f'4090-complete-recovery-{condition}-formal-v1-step{step}'
        original = RESULTS / f'{tag}-original.jsonl'
        receipt = RESULTS / f'{tag}-receipt.json'
        digest = hashlib.sha256(raw).hexdigest()
        if original.exists():
            assert original.read_bytes() == raw, f'Preserved bytes changed: {original}'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert saved['original_record_sha256'] == digest
            assert saved['original_record_bytes'] == len(raw)
            assert original.exists(), 'Receipt without original'
            continue
        selected = record['selected']
        assert len(record['grid']) == 11
        eligible = [p for p in record['grid']
                    if p['ambient_faph'] <= .5 and p['max_source_false_rate'] <= .01]
        # This field is deliberately null for nonzero events, not a zero-event approximation.
        upper = (-math.log(.05) / selected['ambient_hours']
                 if selected['ambient_events'] == 0 else None)
        report = dict(original_path=str(source), preserved_path=str(original),
                      original_record_bytes=len(raw), original_record_sha256=digest,
                      step=step, selected=selected,
                      training_float_eligible_grid_points=len(eligible),
                      returned_point_is_fallback=not bool(eligible),
                      background_zero_event_one_sided95_upper_per_hour=upper,
                      formal_run_complete=False, candidate_promoted=False,
                      limits='Reused float checkpoint only; fallback is not pass, no INT8 or independent acceptance or cross-arm conclusion.')
        if not original.exists():
            with original.open('xb') as handle:
                handle.write(raw)
        with receipt.open('x') as handle:
            handle.write(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'condition': condition, 'preserved_step': step,
                          'sha256': digest, 'eligible': len(eligible)}), flush=True)
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    assert 1 <= args.poll_seconds <= 60
    while True:
        steps = {arm: collect(arm) for arm in ('control', 'recovered')}
        print(json.dumps({'latest_complete_validation_steps': steps,
                          'training_completion_claim': False}), flush=True)
        if not args.watch or all(s == 120000 for s in steps.values()):
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
