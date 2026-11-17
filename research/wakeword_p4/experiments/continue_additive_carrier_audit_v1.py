"""Wait for the existing trainer, then run its independent audit exactly once."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('.local/wakeword/additive-carrier-supervisor-v1')
OUT = Path('research/wakeword_p4/results')


def main():
    # Exclusive creation prevents duplicate audit workers; never launch training.
    work = ROOT / 'audit-continuation'
    work.mkdir(exist_ok=False)
    live = json.loads((ROOT / 'live.json').read_text())
    assert live['arm'] == 'additive'
    pid = live['pid']
    proc = Path('/proc') / str(pid)
    assert b'train_binary40_additive_carrier_pair_v1.py' in (proc / 'cmdline').read_bytes()
    identity = (proc / 'stat').read_text().split()[21]
    while True:
        rows = json.loads((ROOT / 'receipts.json').read_text())
        if len(rows) == 2:
            assert [r['arm'] for r in rows] == ['control', 'additive']
            assert rows[1]['pid'] == pid
            assert all(r['actual_exit_code'] == 0 for r in rows), rows
            break
        try:
            assert (proc / 'stat').read_text().split()[21] == identity
        except FileNotFoundError:
            # Allow the parent to persist the actual wait() result.
            time.sleep(2)
            rows = json.loads((ROOT / 'receipts.json').read_text())
            assert len(rows) == 2, 'Trainer disappeared without terminal receipt'
            continue
        time.sleep(10)
    shutil.copyfile(ROOT / 'receipts.json', OUT / '4090-binary40-additive-carrier-training-receipts-v1.json')
    command = ['/usr/bin/python3', 'research/wakeword_p4/experiments/review_binary40_additive_carrier_training_v1.py']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (work / 'review.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        (work / 'live.json').write_text(json.dumps(dict(pid=child.pid, command=command)) + '\n')
        code = child.wait()
    (work / 'receipt.json').write_text(json.dumps(dict(pid=child.pid, command=command, actual_exit_code=code), indent=2) + '\n')
    (work / 'live.json').unlink()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
