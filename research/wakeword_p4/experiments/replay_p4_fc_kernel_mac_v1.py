"""Replay the installed P4 ANSI FC source on the two saved actual embeddings.

This compiles and executes the actual portable C kernel on Mac, not a P4
interpreter or a qualified candidate. The downstream LiteRT fragment is checked
against the complete original saved output before using its controlled FC input.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(a):
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as s
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    repo = Path(__file__).resolve().parents[3]
    a.output_dir.mkdir(exist_ok=False)
    component = repo / "managed_components/espressif__esp-nn"
    source = component / "src/fully_connected/esp_nn_fully_connected_ansi.c"
    library = a.output_dir / "libp4_fc.dylib"
    command = [shutil.which("clang"), "-std=gnu11", "-O2", "-dynamiclib",
               "-I" + str(component / "src/common"), str(source), "-o", str(library)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (a.output_dir / "compile.json").write_text(json.dumps(vars(compiled), indent=2) + "\n")
    compiled.check_returncode()
    dylib = ctypes.CDLL(str(library))
    fc = dylib.esp_nn_fully_connected_s8_ansi
    p8, p32 = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)
    fc.argtypes = [p8, ctypes.c_int32, ctypes.c_uint16, p8, ctypes.c_int32, p32,
                   p8, ctypes.c_uint16, ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    fc.restype = None
    ref = subprocess.check_output(["git", "rev-parse", a.gpu_ref], text=True).strip()
    raw = subprocess.check_output(["git", "show", ref + ":research/wakeword_p4/results/4090-hey-broad-equalized-export-result-v1.json"])
    export = json.loads(raw)
    prior = json.loads((a.integer_replay_root / "report.json").read_text())
    results = []
    for target in ["hey_willow", "hey_nyx"]:
        arm = next(v for v in export["arms"] if v["target"] == target)
        case = next(v for v in prior["cases"] if v["target"] == target)
        path = a.payload_root / arm["path"]
        assert sha(path) == arm["sha256"]
        assert sha(a.integer_replay_root / case["arrays"]) == case["arrays_sha256"]
        data = path.read_bytes()
        interpreter = Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        interpreter.allocate_tensors()
        details = {v["index"]: v for v in interpreter.get_tensor_details()}
        op, = [o for o in arm["ops"] if o["op"] == "FULLY_CONNECTED"]
        embedding_id, weight_id, bias_id = op["inputs"]
        output_id, = op["outputs"]
        assert op["fused_activation"] == s.ActivationFunctionType.RELU6
        weights = interpreter.get_tensor(weight_id).copy()
        bias = interpreter.get_tensor(bias_id).copy()
        iscale, izero = details[embedding_id]["quantization"]
        wscale, wzero = details[weight_id]["quantization"]
        oscale, ozero = details[output_id]["quantization"]
        assert weights.dtype == np.int8 and bias.dtype == np.int32 and wzero == 0
        channels, depth = weights.shape
        assert channels == 1 and depth == 1800
        assert int(np.max(np.abs(bias.astype(np.int64)))) + depth * 255 * 128 < 1 << 31
        # Installed TFLM scalar preparation multiplies float scales before
        # widening to double (kernel_util.cc scalar overload).
        effective = float(np.float32(np.float32(iscale) * np.float32(wscale))) / oscale
        fraction, shift = math.frexp(effective)
        multiplier = math.floor(fraction * (1 << 31) + .5)
        if multiplier == 1 << 31:
            multiplier //= 2
            shift += 1
        assert -31 <= shift <= 0
        low = max(-128, ozero)
        high = min(127, ozero + math.floor(6 / oscale + .5))
        with np.load(a.integer_replay_root / case["arrays"], allow_pickle=False) as saved:
            embeddings = np.ascontiguousarray(saved[f"tensor_{embedding_id}"].reshape(-1, depth), dtype=np.int8)
            host_fc = saved[f"tensor_{output_id}"].copy().reshape(-1)
            host_final = saved["F"].copy()
        repeats = []
        for _ in range(2):
            output = np.empty((len(embeddings), 1), np.int8)
            for at, row in enumerate(embeddings):
                fc(row.ctypes.data_as(p8), -izero, depth, weights.ctypes.data_as(p8), -wzero,
                   bias.ctypes.data_as(p32), output[at].ctypes.data_as(p8), channels,
                   ozero, shift, multiplier, low, high)
            repeats.append(output.reshape(-1))
        assert np.array_equal(*repeats)

        graph = s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))
        sub = graph.subgraphs[0]
        selected = [n for n, o in enumerate(arm["ops"]) if o["op"] in ["MUL", "ADD", "LOGISTIC", "QUANTIZE"]]
        assert [arm["ops"][n]["op"] for n in selected] == ["MUL", "ADD", "LOGISTIC", "QUANTIZE"]
        sub.operators = [sub.operators[n] for n in selected]
        sub.inputs = np.asarray([output_id], np.int32)
        graph.subgraphs = [sub]
        graph.signatureDefs = []
        builder = flatbuffers.Builder(0)
        builder.Finish(graph.Pack(builder), file_identifier=b"TFL3")
        fragment = bytes(builder.Output())
        fragment_path = a.output_dir / (target + "-post-fc.tflite")
        fragment_path.write_bytes(fragment)
        post = Interpreter(model_content=fragment, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        post.allocate_tensors()
        inp, out = post.get_input_details()[0], post.get_output_details()[0]
        def downstream(values):
            output = []
            for v in values:
                post.set_tensor(inp["index"], np.asarray([[v]], np.int8))
                post.invoke()
                output.append(int(post.get_tensor(out["index"])[0, 0]))
            return np.asarray(output, np.uint8)
        original_codes = downstream(host_fc)
        assert np.array_equal(original_codes.astype(np.float64) / 256, host_final)
        controlled_codes = downstream(repeats[0])
        fc_delta = repeats[0].astype(np.int32) - host_fc.astype(np.int32)
        final_delta = controlled_codes.astype(np.int32) - original_codes.astype(np.int32)
        arrays = a.output_dir / (target + "-fc-replay.npz")
        np.savez_compressed(arrays, host_fc=host_fc, p4_ansi_fc=repeats[0],
                            original_uint8=original_codes, p4_ansi_fc_post_uint8=controlled_codes)
        result = dict(target=target, steps=len(embeddings), actual_c_kernel_repeat_exact=True,
                      host_downstream_fragment_reproduces_original_uint8=True,
                      effective_multiplier=effective, quantized_multiplier=multiplier, shift=shift,
                      fc_unequal_steps=int(np.count_nonzero(fc_delta)), fc_max_abs_code=int(np.max(np.abs(fc_delta))),
                      final_uint8_unequal_steps=int(np.count_nonzero(final_delta)), final_uint8_max_abs_code=int(np.max(np.abs(final_delta))),
                      arrays=arrays.name, arrays_sha256=sha(arrays), fragment_sha256=sha(fragment_path))
        results.append(result)
        print(json.dumps(result), flush=True)
    paths = [source, component / "src/common/common_functions.h", component / "include/esp_nn_esp32p4.h",
             repo / "managed_components/espressif__esp-tflite-micro/tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc",
             repo / "managed_components/espressif__esp-tflite-micro/tensorflow/lite/micro/kernels/fully_connected_common.cc",
             repo / "managed_components/espressif__esp-tflite-micro/tensorflow/lite/kernels/kernel_util.cc",
             repo / "research/panel_wake_runtime/panel_wake_runtime.cpp"]
    report = dict(schema="mac-p4-ansi-fc-kernel-replay-v1", gpu_ref=ref, script_sha256=sha(__file__),
                  source_sha256={str(p.relative_to(repo)): sha(p) for p in paths},
                  export_report_sha256=hashlib.sha256(raw).hexdigest(),
                  original_mac_replay_report_sha256=sha(a.integer_replay_root / "report.json"),
                  library_sha256=sha(library), cases=results,
                  versions={n: importlib.metadata.version(n) for n in ["numpy", "ai-edge-litert"]},
                  limits="Actual installed P4 portable ANSI FC C source executed on Mac using saved real original embeddings. Scalar preparation is source-derived Python, not actual TFLM Prepare. Downstream-only LiteRT fragment reproduces original controls. This isolates FC with host encoder and host postops: not full P4/model execution, a deployed firmware observation, numerical candidate admission, recognition/FAR, or permission to relax exact selftest. Original models unchanged; no PCM IO, optimizer, acceptance or board write.")
    with (a.output_dir / "report.json").open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ref", required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--integer-replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    replay(parser.parse_args())
