"""Compare all learning-rate steps with the pinned upstream implementation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from train_embedding_head import learning_rate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
code = a.upstream.read_bytes()
expected_hash = '8d559e4c1bc9b5523bae50e7ec8f642756a63a7bb5f323620d069a2d03db2972'
assert hashlib.sha256(code).hexdigest() == expected_hash
f = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef)
         and n.name == 'lr_warmup_cosine_decay')
scope = {'np': np}
exec(compile(ast.Module(body=[f], type_ignores=[]), str(a.upstream), 'exec'), scope)
steps = 40000
expected = np.array([scope[f.name](None, i, warmup_steps=steps//5, hold=steps//3,
                                   total_steps=steps, target_lr=1e-4) for i in range(steps)])
actual = np.array([learning_rate(i+1, steps, 'upstream') for i in range(steps)])
np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
report = dict(role='Learning-rate parity, not recognition evidence',
              upstream_revision='368c03716d1e92591906a84949bc477f3a834455',
              upstream_file_sha256=expected_hash, steps=steps,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              bit_exact=bool(np.array_equal(actual, expected)),
              max_absolute_error=float(max(abs(actual-expected))), rtol=1e-14, atol=0,
              comparison_passed=True,
              note='Initial bit-exact assertion failed at machine precision due to warmup arithmetic ordering; training source unchanged.',
              min=float(min(actual)), max=float(max(actual)), regression_or_final_audio_scored=False)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
