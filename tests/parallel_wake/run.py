#!/usr/bin/env python3
"""Build the actual runtime smoke with in-memory constructed inputs only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCES = [
    "tensorflow/compiler/mlir/lite/core/api/error_reporter.cc",
    "tensorflow/compiler/mlir/lite/schema/schema_utils.cc",
    "tensorflow/lite/core/api/flatbuffer_conversions.cc",
    "tensorflow/lite/core/api/tensor_utils.cc",
    "tensorflow/lite/core/c/common.cc",
    "tensorflow/lite/kernels/internal/common.cc",
    "tensorflow/lite/kernels/internal/portable_tensor_utils.cc",
    "tensorflow/lite/kernels/internal/quantization_util.cc",
    "tensorflow/lite/kernels/internal/reference/comparisons.cc",
    "tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc",
    "tensorflow/lite/kernels/internal/tensor_ctypes.cc",
    "tensorflow/lite/kernels/internal/tensor_utils.cc",
    "tensorflow/lite/kernels/kernel_util.cc",
    "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/debug_log.cc",
    "tensorflow/lite/micro/flatbuffer_utils.cc",
    "tensorflow/lite/micro/kernels/assign_variable.cc",
    "tensorflow/lite/micro/kernels/call_once.cc",
    "tensorflow/lite/micro/kernels/concatenation.cc",
    "tensorflow/lite/micro/kernels/conv_common.cc",
    "tensorflow/lite/micro/kernels/depthwise_conv_common.cc",
    "tensorflow/lite/micro/kernels/esp_nn/conv.cc",
    "tensorflow/lite/micro/kernels/esp_nn/depthwise_conv.cc",
    "tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc",
    "tensorflow/lite/micro/kernels/expand_dims.cc",
    "tensorflow/lite/micro/kernels/fully_connected_common.cc",
    "tensorflow/lite/micro/kernels/kernel_util.cc",
    "tensorflow/lite/micro/kernels/logistic.cc",
    "tensorflow/lite/micro/kernels/logistic_common.cc",
    "tensorflow/lite/micro/kernels/read_variable.cc",
    "tensorflow/lite/micro/kernels/reshape.cc",
    "tensorflow/lite/micro/kernels/reshape_common.cc",
    "tensorflow/lite/micro/kernels/strided_slice.cc",
    "tensorflow/lite/micro/kernels/strided_slice_common.cc",
    "tensorflow/lite/micro/kernels/var_handle.cc",
    "tensorflow/lite/micro/memory_helpers.cc",
    "tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc",
    "tensorflow/lite/micro/memory_planner/linear_memory_planner.cc",
    "tensorflow/lite/micro/micro_allocation_info.cc",
    "tensorflow/lite/micro/micro_allocator.cc",
    "tensorflow/lite/micro/micro_context.cc",
    "tensorflow/lite/micro/micro_interpreter.cc",
    "tensorflow/lite/micro/micro_interpreter_context.cc",
    "tensorflow/lite/micro/micro_interpreter_graph.cc",
    "tensorflow/lite/micro/micro_log.cc",
    "tensorflow/lite/micro/micro_op_resolver.cc",
    "tensorflow/lite/micro/micro_profiler.cc",
    "tensorflow/lite/micro/micro_resource_variable.cc",
    "tensorflow/lite/micro/micro_time.cc",
    "tensorflow/lite/micro/micro_utils.cc",
    "tensorflow/lite/micro/recording_micro_allocator.cc",
    "tensorflow/lite/micro/system_setup.cc",
    "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc",
    "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc",
    "tensorflow/lite/micro/kernels/esp_nn/add.cc",
    "tensorflow/lite/micro/kernels/add_common.cc",
    "tensorflow/lite/micro/kernels/quantize.cc",
    "tensorflow/lite/micro/kernels/quantize_common.cc",
    "tensorflow/lite/micro/kernels/split_v.cc"
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--c-compiler',default='cc')
    parser.add_argument('--sanitizer',choices=['undefined','address'],default='undefined')
    parser.add_argument('--frontend',action='store_true',help='Exercise shared frontend, priming/live join and fail-stop lifecycle')
    parser.add_argument('--single',action='store_true',help='Exercise the unchanged default C161 caller and frontend allocation-failure cleanup')
    args=parser.parse_args()
    if not args.single and args.handoff is None:
        parser.error('--handoff is required for the optional parallel detector')
    if args.single:
        args.frontend=True
    root=Path(__file__).resolve().parents[2]
    component=root/'managed_components/espressif__esp-tflite-micro'
    frontend=root/'managed_components/esphome__esp-micro-speech-features'
    args.output.mkdir(parents=True,exist_ok=False)
    sanitize='-fsanitize='+args.sanitizer
    flags=['-std=c++17','-O1','-g',sanitize,'-fno-sanitize-recover=all','-fno-omit-frame-pointer',
           '-DTF_LITE_STATIC_MEMORY','-DTF_LITE_DISABLE_X86_NEON','-DESP_NN=0',
           '-ffp-contract=off','-fno-rtti','-fno-exceptions']
    include=[root/'tests/parallel_wake/shims',root/'components/panel_wake_runtime',
             component,component/'third_party/gemmlowp',component/'third_party/flatbuffers/include',
             component/'third_party/ruy',component/'third_party/kissfft']
    if args.frontend: include += [frontend/'include',frontend/'src']
    if args.single:
        generated=args.output/'generated'
        subprocess.run([sys.executable,str(root/'components/panel_wake_runtime/prepare_streaming_runtime.py'),
                        '--profile',str(root/'models/hey-nyx-c161/profile.json'),
                        '--model',str(root/'models/hey-nyx-c161/model.tflite'),
                        '--output',str(generated)],check=True,capture_output=True,text=True)
        model=(generated/'hey_nyx_model.bin').read_bytes()
        (args.output/'single_model.h').write_text(
            'alignas(16) const uint8_t model_bytes[] asm("_binary_hey_nyx_model_bin_start") = {\n'+
            ','.join(str(value) for value in model)+'\n};\n')
        flags += ['-DPANEL_WAKE_STREAMING_PROFILE=1']
        include += [generated,args.output,root/'managed_components/espressif__esp-sr/include/esp32p4']
    flags += ['-I'+str(p) for p in include]
    sources=[component/p for p in SOURCES]
    if args.frontend: sources += sorted((frontend/'src').glob('*.c'))
    flow='single_frontend_flow.cpp' if args.single else ('frontend_flow.cpp' if args.frontend else 'runtime_flow.cpp')
    sources += [root/'tests/parallel_wake'/flow]
    hash_paths=[*sources,*sorted((root/'components/panel_wake_runtime').glob('*.h')),
                *sorted((root/'tests/parallel_wake/shims').rglob('*.h')),
                *sorted((root/'components/panel_wake_runtime/portable').glob('*.h'))]
    if args.single:
        hash_paths += [root/'components/panel_wake_runtime/panel_wake_runtime.cpp',
                       root/'components/panel_wake_runtime/prepare_streaming_runtime.py',
                       root/'models/hey-nyx-c161/profile.json',args.output/'single_model.h',
                       generated/'wake_config.h']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in hash_paths}
    def compile_one(item):
        i,path=item
        obj=args.output/f'{i}.o'
        if path.suffix=='.c':
            cflags=[flag for flag in flags if flag not in ['-std=c++17','-fno-rtti','-fno-exceptions']]
            # Track only the frontend's explicit C allocations. TFLM and the
            # host C++ library retain their normal allocator behavior.
            cflags += ['-Dmalloc=nyx_test_frontend_malloc',
                       '-Dcalloc=nyx_test_frontend_calloc',
                       '-Dfree=nyx_test_frontend_free']
            command=[args.c_compiler,'-std=c11',*cflags,'-c',str(path),'-o',str(obj)]
        else:
            command=[args.compiler,*flags,'-c',str(path),'-o',str(obj)]
        result=subprocess.run(command,text=True,capture_output=True)
        (args.output/f'{i}.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(f'Compile failed: {path}; see {args.output}/{i}.log')
        return obj
    with ThreadPoolExecutor(max_workers=2) as pool:
        objects=list(pool.map(compile_one,enumerate(sources)))
    executable=args.output/'runtime-flow'
    subprocess.run([args.compiler,sanitize,*[str(p) for p in objects],'-o',str(executable)],check=True)
    models=([root/'models/hey-nyx-c161/model.tflite'] if args.single else
            [args.handoff/'models'/f'{name}.tflite' for name in ['c161','binary40','history_expert']])
    command=[str(executable)]+([] if args.single else [str(p) for p in models])
    timed_out=False
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    except subprocess.TimeoutExpired as error:
        timed_out=True
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        result=subprocess.CompletedProcess(command,124,decoded(error.stdout),
                    decoded(error.stderr)+'\nHost smoke exceeded 30 seconds; child terminated.\n')
    (args.output/'runtime.log').write_text(result.stdout+result.stderr)
    report={'schema':'parallel-runtime-host-flow-v1','exit_code':result.returncode,'timed_out':timed_out,
            'source_sha256':hashes,'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in models},
            'compiler':subprocess.check_output([args.compiler,'--version'],text=True).splitlines()[0],
            'flags':flags,'sanitizer':args.sanitizer,'frontend_exercised':args.frontend,
            'default_single_c161_caller_exercised':args.single,
            'input_kind':'in-memory constructed PCM' if args.frontend else 'constructed feature chunks',
            'new_training_tts_playback_or_recording':False,
            'scope':'Actual host TFLM allocation, ABI/cache ownership, invocation and failure cleanup; optional actual shared frontend/prime-live continuity. Not P4 recognition, memory peaks or acoustic acceptance.'}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout[-4000:],flush=True)
    if result.returncode: print(result.stderr[-4000:],flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
