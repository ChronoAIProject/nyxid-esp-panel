"""Prepare a standalone P4 project from unchanged original trained witnesses.

No model invocation, frontend, microphone, network or hardware access occurs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np


BASE = Path("research/wakeword_p4")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def prepare(repo, packet, output, public_plan):
    assert not output.exists() and not public_plan.exists()
    intake_path = repo / BASE / "results/mac-fp32-internal-fullstream-intake-v1.json"
    intake = json.loads(intake_path.read_bytes())
    manifest_bytes = (packet / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    # Authenticate every member actually used against the already admitted
    # original packet manifest; do not accept arbitrary replacement witnesses.
    assert sha(manifest_bytes) == intake["manifest_sha256"]

    def member(name):
        data = (packet / name).read_bytes()
        assert len(data) == manifest["files"][name]["bytes"]
        assert sha(data) == manifest["files"][name]["sha256"]
        return data

    lock_name = "receipts/4090-fp32-internal-fullstream-execution-v1.json"
    lock_bytes = member(lock_name)
    lock = json.loads(lock_bytes)
    requests = lock["requests"]
    assert len(requests) == 135
    # Selection depends only on original order/engine/word/variant, never scores.
    selected = []
    for target in ("hey_willow", "hey_nyx"):
        for engine in ("piper", "kokoro", "qwen"):
            index = next(i for i, r in enumerate(requests)
                         if r["variant"] == 0 and r["source"]["target"] == target
                         and r["source"]["engine"] == engine)
            selected.append(index)
    assert len(set(selected)) == 6
    output.mkdir()
    project = output / "project"
    main = project / "main"
    main.mkdir(parents=True)
    assets = {}

    def asset(name, data):
        (main / name).write_bytes(data)
        assets[name] = dict(bytes=len(data), sha256=sha(data))
        return name

    exports = lock["exports"]
    assert len(exports) == 4
    declarations = ["#pragma once", "#include <cstddef>", "#include <cstdint>",
                    "struct Model { const uint8_t *data; };",
                    "struct Stream { const uint8_t *input; size_t steps; };",
                    "struct Case { unsigned model, stream; const uint8_t *expected_tf; };"]

    def embed(name):
        symbol = "asset_" + str(len(declarations))
        binary = name.replace(".", "_")
        declarations.append(f'extern const uint8_t {symbol}[] asm("_binary_{binary}_start");')
        return symbol

    models = []
    model_symbols = []
    streams = []
    cases = []
    stream_symbols = []
    for mi, exported in enumerate(exports):
        name = exported["target"] + "-" + exported["condition"]
        original_member = "models/" + name + ".tflite"
        data = member(original_member)
        assert sha(data) == exported["sha256"]
        filename = asset(f"model_{mi}.tflite", data)
        model_symbols.append(embed(filename))
        models.append(dict(name=name, original_member=original_member, embedded_file=filename,
                           bytes=len(data), sha256=sha(data)))
        for si, index in enumerate(selected):
            original_npz = f"arrays/{name}-{index:03}.npz"
            npz_data = member(original_npz)
            with np.load(io.BytesIO(npz_data), allow_pickle=False) as z:
                values = z["input"]
                tf = z["tf"]
                litert = z["litert"]
                assert values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 40
                assert len(values) % 3 == 0 and tf.shape == litert.shape == (len(values) // 3,)
                assert all(np.isfinite(a).all() for a in (values, tf, litert))
                assert tf.tobytes() == z["tf_repeat"].tobytes()
                assert litert.tobytes() == z["litert_repeat"].tobytes()
                data = values.astype("<f4").tobytes()
                expected_tf = tf.astype("<f4").tobytes()
                expected_litert = litert.astype("<f4").tobytes()
            steps = len(data) // 480
            if mi == 0:
                filename = asset(f"stream_{si}.bin", data)
                stream_symbols.append(embed(filename))
                r = requests[index]
                streams.append(dict(index=index, request_order=r["order"], target=r["source"]["target"],
                                    engine=r["source"]["engine"], source_id=r["source"]["id"],
                                    variant=r["variant"], steps=steps, input_file=filename,
                                    input_sha256=sha(data)))
            else:
                assert sha(data) == streams[si]["input_sha256"] and steps == streams[si]["steps"]
            tf_name = asset(f"tf_{mi}_{si}.bin", expected_tf)
            tf_symbol = embed(tf_name)
            lite_name = f"litert_{mi}_{si}.bin"
            (output / lite_name).write_bytes(expected_litert)
            cases.append(dict(case=len(cases), model=mi, stream=si, original_npz=original_npz,
                              original_npz_sha256=sha(npz_data), steps=steps,
                              tf_file="project/main/" + tf_name, tf_sha256=sha(expected_tf),
                              litert_file=lite_name, litert_sha256=sha(expected_litert),
                              tf_symbol=tf_symbol))
    declarations.append("static const Model kModels[] = {" + ",".join("{" + s + "}" for s in model_symbols) + "};")
    declarations.append("static const Stream kStreams[] = {" + ",".join("{" + s + "," + str(r["steps"]) + "}" for s, r in zip(stream_symbols, streams, strict=True)) + "};")
    declarations.append("static const Case kCases[] = {" + ",".join("{" + str(r["model"]) + "," + str(r["stream"]) + "," + r["tf_symbol"] + "}" for r in cases) + "};")
    declarations.append("static constexpr size_t kCaseCount = sizeof(kCases)/sizeof(kCases[0]);")
    (main / "fixtures.h").write_text("\n".join(declarations) + "\n")
    probe = Path(__file__).with_name("fp32_trained_fullstream_p4_v1.cpp")
    shutil.copyfile(probe, main / "probe.cpp")
    frozen = json.loads((repo / BASE / "results/mac-fp32-runtime-feasibility-plan-v1.json").read_bytes())
    components = {}
    for name, expected in frozen["identity"]["components"].items():
        source = repo / "managed_components" / ("espressif__" + name)
        tree = {str(p.relative_to(source)): sha(p.read_bytes()) for p in source.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(source).parts}
        assert len(tree) == expected["files"] and sha(json.dumps(tree, sort_keys=True).encode()) == expected["sha256"]
        shutil.copytree(source, project / "components" / name)
        components[name] = expected
    cmake = project / "components/esp-tflite-micro/CMakeLists.txt"
    before = cmake.read_text()
    assert before.count("set(priv_req)") == 1
    cmake.write_text(before.replace("set(priv_req)", "set(pub_req esp-nn)\n\nset(priv_req)"))
    (project / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.16)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\nproject(nyx_trained_fp32_probe)\n')
    embed_names = " ".join('"' + name + '"' for name in assets)
    (main / "CMakeLists.txt").write_text('idf_component_register(SRCS "probe.cpp" INCLUDE_DIRS "." REQUIRES esp-tflite-micro esp-nn esp_timer EMBED_FILES ' + embed_names + ')\n')
    defaults = (repo / BASE / "sdkconfig.defaults").read_text()
    defaults += '\nCONFIG_PARTITION_TABLE_CUSTOM=y\nCONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"\n'
    (project / "sdkconfig.defaults").write_text(defaults)
    (project / "partitions.csv").write_text("# Build layout only; never flash this partition table.\nnvs,data,nvs,0x9000,0x6000\nphy_init,data,phy,0xf000,0x1000\nfactory,app,factory,0x10000,0x400000\n")
    generated = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
                 for p in output.rglob("*") if p.is_file() and "components" not in p.relative_to(output).parts}
    plan = dict(schema="mac-fp32-trained-fullstream-p4-preparation-v1", source_sha256=sha(Path(__file__).read_bytes()),
                probe_sha256=sha(probe.read_bytes()), intake_sha256=sha(intake_path.read_bytes()),
                original_packet_manifest_sha256=sha(manifest_bytes), original_execution_sha256=sha(lock_bytes),
                selection="First original variant0 per target in Willow/Nyx order, engine Piper/Kokoro/Qwen order; no score selection; complete stream including startup and tail.",
                models=models, streams=streams, cases=cases, repeats=2,
                expected_invocations=2 * sum(c["steps"] for c in cases),
                assets=assets, generated_files=generated, components=components,
                max_abs=1e-4, every_invoke_deadline_us=30000,
                actual_model_invocations=0, hardware_accessed=False, candidate_qualified=False,
                limits="Prepared original trained numeric/resource subset only, not full540, recognition, real P4 execution or full-panel headroom. Raw float reconstruction of integer-trained weights remains distinct from quantized training execution. No model re-export, frontend, audio, fresh development or acceptance access. Future app-only run requires fresh development-board identity/slot/backup and finally restore; never flash generated partition table/bootloader or access dining-room production board.")
    write_json(output / "fixture-plan.json", plan)
    write_json(public_plan, plan)
    print(json.dumps({k: plan[k] for k in ("schema", "expected_invocations", "actual_model_invocations", "hardware_accessed")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "packet-root", "output-dir", "plan"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.repository_root.resolve(), a.packet_root.resolve(), a.output_dir.resolve(), a.plan.resolve())


if __name__ == "__main__":
    main()
