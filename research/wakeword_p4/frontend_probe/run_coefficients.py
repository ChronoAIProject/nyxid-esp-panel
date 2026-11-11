"""Compile a numeric-only MicroFrontend table dump in an isolated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True,
                   help='Unmodified pymicro-features2.0.2 source root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compiler', default='c++')
    p.add_argument('--arch', choices=('arm64', 'x86_64'),
                   help='Optional macOS cross-build/run; requires native support or Rosetta')
    p.add_argument('--fp-contract', choices=('on', 'off'),
                   help='Diagnostic alternate: explicitly control fused floating arithmetic')
    args = p.parse_args()
    source = args.source.resolve()
    own = Path(__file__).with_name('coefficients.cpp').resolve()
    lib = source/'tensorflow/lite/experimental/microfrontend/lib'
    units = [own, lib/'filterbank_util.cc', lib/'window_util.cc']
    inputs = sorted({*lib.glob('*.h'), *units[1:]})
    if any(not path.is_file() for path in units):
        p.error('Missing original C++ frontend source files')
    args.output.mkdir(parents=True, exist_ok=False)
    executable = args.output.resolve()/'coefficients'
    command = [args.compiler, '-std=c++17', '-O2', '-fno-fast-math']
    if args.fp_contract:
        command.append('-ffp-contract='+args.fp_contract)
    if args.arch:
        if platform.system() != 'Darwin':
            p.error('--arch applies only to macOS')
        command += ['-arch', args.arch]
    command += ['-I', str(source), *map(str, units), '-o', str(executable)]
    built = subprocess.run(command, capture_output=True, text=True)
    (args.output/'build.log').write_text(built.stdout+built.stderr)
    built.check_returncode()
    run = (['arch', '-'+args.arch] if args.arch else [])+[str(executable)]
    result = subprocess.run(run, capture_output=True)
    (args.output/'stderr.log').write_bytes(result.stderr)
    result.check_returncode()
    tables = json.loads(result.stdout)
    assert (tables['channels'], tables['window_step']) == (40, 160)
    assert len(tables['window_coefficients']) == 480
    assert len(tables['weights']) == len(tables['unweights'])
    (args.output/'tables.json').write_bytes(result.stdout)
    report = {
        'scope': 'Numeric-only upstream filterbank/window table diagnostic; no input PCM or model.',
        'platform': platform.platform(), 'python': sys.version, 'arch': args.arch,
        'fp_contract': args.fp_contract,
        'compiler_version': subprocess.check_output([args.compiler, '--version'], text=True),
        'command': command, 'run_command': run,
        'source_sha256': {str(path.relative_to(source)): sha(path) for path in inputs},
        'probe_sha256': sha(own), 'runner_sha256': sha(Path(__file__)),
        'executable_sha256': sha(executable), 'tables_sha256': sha(args.output/'tables.json'),
        'audio_io': False, 'board_access': False, 'training_or_inference': False,
        'limits': 'Standalone compiler does not prove installed-wheel coefficients or P4 agreement. FFT/PCAN/runtime stages are not inspected.',
    }
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'tables_sha256': report['tables_sha256'], 'arch': args.arch}))


if __name__ == '__main__':
    main()
