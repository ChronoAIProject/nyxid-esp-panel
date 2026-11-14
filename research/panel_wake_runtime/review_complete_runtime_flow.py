"""Compile actual detector code against host doubles and audit finite stream flow.

This does not execute MicroFrontend FFT, TFLite operators or ESP32-P4 hardware.
Output directories are immutable so failed attempts are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EMPTY_HEADERS = ['esp_wn_iface.h', 'esp_heap_caps.h', 'esp_log.h', 'esp_timer.h',
    'freertos/FreeRTOS.h', 'freertos/task.h', 'frontend.h', 'frontend_util.h',
    'tensorflow/lite/micro/micro_interpreter.h', 'tensorflow/lite/micro/micro_mutable_op_resolver.h',
    'tensorflow/lite/micro/micro_resource_variable.h', 'tensorflow/lite/schema/schema_generated.h']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(output, legacy_ref, sanitizers):
    output.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('clang++ is required for sanitizer host checks')
    includes = output/'include'
    for name in EMPTY_HEADERS:
        path = includes/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (includes/'wake_config.h').write_text('#pragma once\n#define PANEL_WAKE_MODEL_SHA256 "constructed"\n'
        '#define PANEL_WAKE_THRESHOLD .55\n#define PANEL_WAKE_SMOOTHING_STEPS 5\n'
        '#define PANEL_WAKE_PRIME_SAMPLES 68000\n#define PANEL_WAKE_SELFTEST_SHA256 "constructed"\n')
    current = (HERE/'panel_wake_runtime.cpp').read_bytes()
    original = subprocess.check_output(['git', 'show', legacy_ref+':research/panel_wake_runtime/panel_wake_runtime.cpp'], cwd=HERE)
    results = {}

    def build(name, source, flags=(), fixture=None):
        root = output/name
        root.mkdir()
        (root/'panel_wake_runtime.cpp').write_bytes(source)
        shutil.copyfile(HERE/'test_complete_runtime_flow.cpp', root/'flow.cpp')
        command = [compiler, '-std=c++20', '-O1', '-g', '-fsanitize='+sanitizers, '-fno-sanitize-recover=all',
            '-fno-omit-frame-pointer', '-include', str(HERE/'test_complete_runtime_shim.h'),
            '-I'+str(includes), '-I'+str(HERE), *flags, str(root/'flow.cpp')]
        if fixture is not None:
            path = root/'fixture.bin'
            path.write_bytes(fixture)
            section = '.section __DATA,__const' if sys.platform == 'darwin' else '.section .rodata'
            asm = root/'fixture.S'
            asm.write_text(section+'\n.balign 4\n.global _binary_wake_selftest_bin_start\n'
                '_binary_wake_selftest_bin_start:\n.incbin "'+str(path)+'"\n'
                '.global _binary_wake_selftest_bin_end\n_binary_wake_selftest_bin_end:\n')
            # The forced C++ header must not be passed to the assembler.
            asm_result = subprocess.run([compiler, '-c', str(asm), '-o', str(root/'fixture.o')], capture_output=True, text=True)
            (root/'asm.json').write_text(json.dumps(vars(asm_result), indent=2))
            if asm_result.returncode:
                raise RuntimeError('Fixture assembly failed: '+name)
            command.append(str(root/'fixture.o'))
        command += ['-o', str(root/'probe')]
        compiled = subprocess.run(command, capture_output=True, text=True)
        (root/'compile.json').write_text(json.dumps(vars(compiled), indent=2))
        if compiled.returncode:
            raise RuntimeError('Host compile failed: '+name+'; see '+str(root/'compile.json'))
        executed = subprocess.run([str(root/'probe')], capture_output=True, text=True)
        (root/'execution.json').write_text(json.dumps(vars(executed), indent=2))
        if executed.returncode:
            raise RuntimeError('Host flow failed: '+name+'; see '+str(root/'execution.json'))
        result = json.loads(executed.stdout)
        results[name] = result
        return result

    complete = build('complete', current, ['-DPANEL_WAKE_COMPLETE_INPUT=1'])
    legacy = build('legacy', current)
    original_result = build('original', original, ['-DFLOW_LEGACY_ORIGINAL=1'])
    for key in ('groups', 'last', 'armed'):
        if not complete[key] == legacy[key] == original_result[key]:
            raise AssertionError('Continuous prefix differs from original: '+key)
    if complete['stream_cases'] != 27 or legacy['stream_cases'] != 27:
        raise AssertionError('Missing stream cases')

    raw = 0
    for _ in range(120):
        raw = (raw*17+166)%256  # int8 -90 represented as UINT8.

    def fixture(samples=960, count=2, magic=b'NYXKWS02', trace=None):
        return magic+bytes(32)+struct.pack('<II', samples, count)+bytes(samples*2)+(bytes([raw])*count if trace is None else trace)

    cases = {'valid-v2': (fixture(), True), 'short-header': (bytes(47), False),
        'short-payload': (fixture()[:-1], False), 'wrong-count': (fixture(count=1), False),
        'wrong-magic': (fixture(magic=b'NYXKWS01'), False),
        'oversized-count': (b'NYXKWS02'+bytes(32)+struct.pack('<II', 0xffffffff, 1), False),
        'wrong-first-output': (fixture(trace=bytes([(raw+1)%256, raw])), False),
        'wrong-final-output': (fixture(trace=bytes([raw, (raw+1)%256])), False)}
    for name, (data, expected) in cases.items():
        build(name, current, ['-DPANEL_WAKE_COMPLETE_INPUT=1', '-DPANEL_WAKE_SELFTEST=1',
            '-DFLOW_SELFTEST_CASE=1', '-DFLOW_EXPECT_SELFTEST_PASS='+str(int(expected))], data)
    build('valid-v1', current, ['-DPANEL_WAKE_SELFTEST=1', '-DFLOW_SELFTEST_CASE=1',
        '-DFLOW_EXPECT_SELFTEST_PASS=1'], fixture(count=1, magic=b'NYXKWS01'))
    result = dict(schema='complete-runtime-host-flow-review-v1',
        current_runtime_sha256=sha(current), original_ref=legacy_ref, original_runtime_sha256=sha(original),
        harness_sha256=sha((HERE/'test_complete_runtime_flow.cpp').read_bytes()),
        shim_sha256=sha((HERE/'test_complete_runtime_shim.h').read_bytes()),
        complete_stream_cases=27, legacy_stream_cases=27,
        continuous_prefix_outputs=len(complete['groups']),
        original_legacy_complete_prefix_exact=True, finite_selftest_cases=len(cases), legacy_v1_selftest_pass=True,
        sanitizers=sanitizers, all_passed=True,
        limits='Actual runtime control flow with constructed frontend/interpreter doubles. No FFT, learned model, ESP hardware, audio IO, recognition, numerical model parity or resource acceptance.')
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-ref', required=True)
    parser.add_argument('--sanitizers', choices=['undefined,bounds', 'address,undefined'], default='undefined,bounds')
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), args.legacy_ref, args.sanitizers), indent=2))
