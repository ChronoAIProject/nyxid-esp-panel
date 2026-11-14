"""Saved GPU state/optimizer audit only, no network or gradient reexecution."""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path('research/wakeword_p4/results')
BASE = Path('.local/wakeword/multispeaker-gpu-smoke-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report_path = R / '4090-multispeaker-gpu-smoke-v1.json'
    report = json.loads(report_path.read_text())
    plan_path = R / '4090-multispeaker-gpu-smoke-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == report['plan_sha256'] and report['all_update_checks_passed']
    inventory = plan['initial']['variable_inventory']['variables']
    arms = []
    for item in report['arms']:
        arm = item['arm']
        root = BASE / arm
        for name, digest in item['files'].items():
            assert sha(root / name) == digest
        assert [s['optimizer_iteration'] for s in item['steps']] == [1, 2]
        assert all('GPU:0' in s['logits_device'] and np.isfinite(s['loss']) for s in item['steps'])
        assert [s['x_sha256'] for s in item['steps']] == [s['x_sha256'] for s in plan['batches'][arm]]
        changed = []
        with np.load(root / 'initial-model-variables.npz', allow_pickle=False) as before, np.load(root / 'final-model-variables.npz', allow_pickle=False) as after:
            assert set(before.files) == set(after.files) == {e['key'] for e in inventory}
            for e in inventory:
                a, b = before[e['key']], after[e['key']]
                assert hashlib.sha256(a.tobytes()).hexdigest() == e['sha256']
                assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                if not e['effective_trainable']:
                    assert a.tobytes() == b.tobytes()
                elif a.tobytes() != b.tobytes():
                    changed.append(e['index'])
        assert changed
        with np.load(root / 'initial-optimizer.npz', allow_pickle=False) as before, np.load(root / 'final-optimizer.npz', allow_pickle=False) as after:
            assert len(before.files) == len(after.files) == 32
            assert before['v000'].item() == 0 and after['v000'].item() == 2
            assert before['v001'].item() == after['v001'].item() == np.float32(1e-5)
            assert all(np.all(before[f'v{i:03}'] == 0) for i in range(2, 32))
            assert all(np.isfinite(after[k]).all() for k in after.files)
            assert any(np.any(after[f'v{i:03}'] != 0) for i in range(2, 32))
        arms.append(dict(arm=arm, preserved_initial31variables_match=True,
                         frozen16variables_unchanged=True, changed_trainable_indices=changed,
                         optimizer_initial_slots_zero=True, optimizer_final_iteration=2,
                         optimizer_variables=32))
    assert {a['arm'] for a in arms} == {'narrow', 'broad'}
    result = dict(schema='multispeaker-gpu-smoke-saved-state-audit-v1',
                  script_sha256=sha(__file__), original_report_sha256=sha(report_path), arms=arms,
                  all_saved_state_checks_passed=True, model_calls=0,
                  limits='Saved actual initial/final model and Adam arrays, full artifact hashes and original runtime receipts only. No independent autodiff or GPU execution, cross-device equality, streaming export, recognition or candidate qualification.')
    with (R / '4090-multispeaker-gpu-smoke-saved-state-audit-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
