"""Compile three runtime variants using an existing real P4 compile database.

No CMake/full-app link, model allocation or hardware operation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def run(database, output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().with_name('panel_wake_runtime.cpp')
    row = next(row for row in json.loads(database.read_bytes()) if row['file'].endswith('panel_wake_runtime.cpp'))
    original = shlex.split(row['command'])
    version = subprocess.check_output([original[0], '--version'], text=True).splitlines()[0]
    variants = []
    for name, complete, selftest in [('complete-selftest', True, True),
            ('legacy-selftest', False, True), ('legacy-continuous', False, False)]:
        build = output/name
        build.mkdir()
        header = ('#pragma once\n#define PANEL_WAKE_MODEL_SHA256 "constructed-compilation-only"\n'
            '#define PANEL_WAKE_THRESHOLD .98828125\n#define PANEL_WAKE_SMOOTHING_STEPS 5\n'
            '#define PANEL_WAKE_PRIME_SAMPLES 68000\n')
        if complete:
            header += '#define PANEL_WAKE_COMPLETE_INPUT 1\n#define PANEL_WAKE_SELFTEST_VERSION 2\n'
        (build/'wake_config.h').write_text(header)
        command = [arg for arg in original if arg != '-DPANEL_WAKE_SELFTEST=1']
        command[command.index('-o')+1] = str(build/'runtime.o')
        command[command.index('-c')+1] = str(source)
        command[1:1] = ['-I'+str(build), '-I'+str(source.parent)]+(['-DPANEL_WAKE_SELFTEST=1'] if selftest else [])
        result = subprocess.run(command, cwd=row['directory'], capture_output=True, text=True)
        (build/'compile.json').write_text(json.dumps(vars(result), indent=2)+'\n')
        variants.append(dict(name=name, exit_code=result.returncode,
            warning_count=result.stderr.count('warning:'),
            object_bytes=(build/'runtime.o').stat().st_size if result.returncode == 0 else None))
        if result.returncode:
            raise RuntimeError('Cross compile failed; inspect '+str(build/'compile.json'))
    report = dict(schema='complete-runtime-p4-cross-compile-v1', toolchain=version,
        runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        real_compile_database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
        variants=variants, all_passed=True,
        limits='Real ESP32-P4 translation-unit compilation with installed ESP-SR, TFLM and MicroFrontend headers. Not a CMake/full-app link, model allocation, P4 execution or recognition test.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.compile_database.resolve(), args.output.resolve()), indent=2))
