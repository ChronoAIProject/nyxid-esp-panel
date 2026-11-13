"""Additive exact-source adapter for fresh Jenny official reference streams."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('research/wakeword_p4')
PLAN = ROOT/'results/4090-jenny-official-development-execution-v1.json'
ORIGINAL = Path(__file__).with_name('run_development.py')


def main():
    lock = json.loads(PLAN.read_text())
    expected = lock['files'][str(ORIGINAL)]
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == expected
    tree = ast.parse(ORIGINAL.read_text())
    matches = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'research/wakeword_p4/results/4090-hey-official-development-execution-v1.json':
            node.value = str(PLAN)
            matches += 1
    assert matches == 1
    # Original admission checks now authenticate this new adapter's __file__
    # plus the separately bound original source and new published plan.
    exec(compile(ast.fix_missing_locations(tree), str(ORIGINAL), 'exec'),
         dict(__name__='__main__', __file__=__file__))


if __name__ == '__main__':
    main()
