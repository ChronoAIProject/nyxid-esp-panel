"""Check actual P4 compile flags, opt-in guard and original OFF C behavior.

Requires the isolated two-engine long-stream build tree. No device/model I/O.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[3]
ORIGINAL = '00b36d4'
SOURCE = 'research/wakeword_baseline/main/baseline.c'
CMAKE = 'research/wakeword_baseline/main/CMakeLists.txt'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def flags(entry):
    tokens = shlex.split(entry['command']); result = []; index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ('-o', '-MF', '-MT', '-MQ'):
            index += 2; continue
        if (token in ('-c', '-MD', '-MMD', '-MP') or token == entry['file']
                or token.startswith(('-DBASELINE_LONG_STREAMS=', '-DBASELINE_THRESHOLD_CONTROL=', '-DBASELINE_REPORT_RESOURCES='))):
            index += 1; continue
        result.append(token); index += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    original = subprocess.check_output(['git', 'show', ORIGINAL + ':' + SOURCE], cwd=REPO)
    current = (REPO / SOURCE).read_bytes()
    assert current == (args.build_root / 'project' / SOURCE).read_bytes()
    (args.output / 'original.c').write_bytes(original)
    (args.output / 'current.c').write_bytes(current)
    # A common temporary filename excludes __FILE__ path differences. Only the
    # four ESP_ERROR_CHECK diagnostic __LINE__ arguments may be normalized.
    stable = args.output / 'comparison-input.c'
    diagnostic = re.compile(r'(_esp_error_check_failed\(err_rc_, "[^"]+", )\d+(, __func__,)')
    comparisons = []; on_compiles = []; guards = []
    for selection in ('hiesp', 'willow'):
        build = args.build_root / ('build-' + selection)
        entry = next(e for e in json.loads((build / 'compile_commands.json').read_text()) if e['file'].endswith('/main/baseline.c'))
        base = flags(entry)
        for control in (0, 1):
            for resources in (0, 1):
                prefix = f'{selection}-control{control}-resources{resources}'
                defines = [f'-DBASELINE_THRESHOLD_CONTROL={control}', f'-DBASELINE_REPORT_RESOURCES={resources}', '-DBASELINE_LONG_STREAMS=0']
                versions = []
                for label, data in (('original', original), ('current', current)):
                    stable.write_bytes(data)
                    result = subprocess.run([*base, *defines, '-E', '-P', str(stable)], cwd=build, capture_output=True)
                    result.check_returncode()
                    (args.output / (prefix + '-' + label + '.i')).write_bytes(result.stdout)
                    versions.append(result.stdout.decode())
                canonical = [diagnostic.subn(r'\g<1>0\g<2>', v) for v in versions]
                assert canonical[0][1] == canonical[1][1] == 4
                assert canonical[0][0] == canonical[1][0]
                diff = '\n'.join(difflib.unified_diff(versions[0].splitlines(), versions[1].splitlines(), n=0)) + '\n'
                (args.output / (prefix + '.diff')).write_text(diff)
                output = args.output / (prefix + '.o')
                result = subprocess.run([*base, *defines, '-c', str(stable), '-o', str(output)], cwd=build, capture_output=True)
                (args.output / (prefix + '-compile.txt')).write_bytes(result.stdout + result.stderr)
                result.check_returncode(); assert b'warning:' not in result.stderr
                comparisons.append(dict(engine=selection, threshold_control=bool(control), resources=bool(resources),
                    long_streams=False, original_preprocessed_sha256=sha(versions[0].encode()),
                    current_preprocessed_sha256=sha(versions[1].encode()),
                    differences_only_four_ESP_ERROR_CHECK_diagnostic_line_constants=True,
                    current_off_branch_compiled=True, compiler_warning_count=0, object_sha256=sha(output.read_bytes())))
        for resources in (0, 1):
            output = args.output / f'{selection}-long-resources{resources}.o'
            result = subprocess.run([*base, '-DBASELINE_THRESHOLD_CONTROL=1', f'-DBASELINE_REPORT_RESOURCES={resources}',
                '-DBASELINE_LONG_STREAMS=1', '-c', str(stable), '-o', str(output)], cwd=build, capture_output=True)
            result.check_returncode(); assert b'warning:' not in result.stderr
            on_compiles.append(dict(engine=selection, resources=bool(resources), long_streams=True,
                compiler_warning_count=0, object_sha256=sha(output.read_bytes())))
        invalid = subprocess.run([*base, '-DBASELINE_THRESHOLD_CONTROL=0', '-DBASELINE_REPORT_RESOURCES=0',
            '-DBASELINE_LONG_STREAMS=1', '-E', str(stable)], cwd=build, capture_output=True)
        assert invalid.returncode != 0 and b'Long streams require the explicit threshold and complete-event protocol' in invalid.stderr
        guards.append(dict(engine=selection, unsupported_compile_profile_rejected=True))
    cmake = (REPO / CMAKE).read_text()
    block = cmake[cmake.index('option(BASELINE_REPORT_RESOURCES'):cmake.index('set_property(CACHE BASELINE_WAKE_MODEL')]
    script = args.output / 'actual-cmake-option-guard.cmake'; script.write_text(block)
    for control, long, valid in [('OFF', 'OFF', True), ('ON', 'OFF', True), ('ON', 'ON', True), ('OFF', 'ON', False)]:
        result = subprocess.run([shutil.which('cmake'), '-DBASELINE_THRESHOLD_CONTROL=' + control,
            '-DBASELINE_LONG_STREAMS=' + long, '-P', str(script)], capture_output=True)
        assert (result.returncode == 0) is valid
        if not valid:
            assert b'BASELINE_LONG_STREAMS requires BASELINE_THRESHOLD_CONTROL' in result.stderr
    report = dict(schema='official-long-stream-compile-check-v1', original_commit=ORIGINAL,
        original_source_sha256=sha(original), current_source_sha256=sha(current),
        checker_sha256=sha(Path(__file__).read_bytes()), off_comparisons=comparisons,
        long_on_compilations=on_compiles, compile_guards=guards, actual_cmake_option_guard_four_profiles_checked=True,
        scope='Actual P4 preprocessing/compilation and option-guard checks. OFF differs only in four diagnostic line constants; not whole-app binary byte identity or native long-stream acceptance.',
        native_inference=False, device_operated=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(off_comparisons=len(comparisons), long_on_compiles=len(on_compiles),
        invalid_compile_profiles_rejected=len(guards), cmake_guard_profiles=4, compiler_warnings=0)))


if __name__ == '__main__':
    main()
