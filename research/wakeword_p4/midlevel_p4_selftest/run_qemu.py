"""Check two P4 TFLM interpreters against every native forward proposal and score."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from fit_c161_history_gain_v1 import RESULTS, read, sha
from run_c161_first_command_v5 import boot
from run_c161_coded_cue_p4_v5 import QEMU
from make_fixture import WORK

STEM = "4090-c161-midlevel-p4-qemu"


def main():
    build = WORK / "build"
    uart = WORK / "uart.log"
    boot(build=build, log=uart, image=WORK / "flash.bin", marker=b"MIDLEVEL_DONE")
    text = uart.read_text(errors="replace")
    match = re.search(r"MIDLEVEL_(PASS|FAIL) blocks=(\d+) calls=(\d+),(\d+) score_errors=(\d+) event_errors=(\d+) "
                      r"tensors=(\d+),(\d+) variables=(\d+),(\d+) max_us=(\d+) over30ms=(\d+) min_internal=(\d+)", text)
    assert match, text[-1800:]
    status = match[1]
    blocks, normal, expanded, errors, event_errors, t0, t1, v0, v1, max_us, over, heap = map(int, match.groups()[1:])
    differences = [[int(v) for v in row] for row in re.findall(
        r"MIDLEVEL_DIFF source=(\d+) step=(\d+) lane=(\d+) expected=(\d+) actual=(\d+)", text)]
    assert len(differences) == errors
    observed = [[int(i) for i in group] for group in re.findall(
        r"MIDLEVEL_STREAM source=(\d+) blocks=(\d+) proposals=(\d+),(\d+)", text)]
    ref = RESULTS / "4090-c161-stream-stretch-forward-v1.json"
    rows = [r for r in read(ref)["records"] if r["role"] == "nyx"]
    assert observed == [[int(r["key"]), r["physical_calls"], len(r["events"]["midlevel"]["identity"]),
                         len(r["events"]["midlevel"]["stretch"])] for r in rows]
    assert len(rows) == 12 and normal == blocks == sum(r["physical_calls"] for r in rows)
    assert expanded == sum(r["expanded_calls"] for r in rows)
    fixture = WORK / "midlevel.bin"
    assert sha(fixture) == "9730756e917a32074e30507f7857ce96fe6758ed543853a895acc3873db4df55"
    cache = (build / "CMakeCache.txt").read_text()
    setting = re.search(r"^NYX_REFERENCE_OPS:[^=]+=(.*)$", cache, re.M)
    ops = setting[1].split(";") if setting and setting[1] else []
    stem = STEM + ("-" + "-".join(ops) if ops else "-ansi")
    trace = "NYX_TRACE_TENSORS:BOOL=ON" in cache
    single = "NYX_FC_SINGLE_ROUNDING:BOOL=ON" in cache
    if single:
        stem += "-fc-single-round"
    if trace:
        stem += "-trace"
    output = WORK / "results"
    output.mkdir(exist_ok=True)
    log = output / f"{stem}-uart.log"
    log.write_bytes(uart.read_bytes())
    files = read(build / "flasher_args.json")["flash_files"]
    config = (WORK / "sdkconfig").read_text()
    assert "CONFIG_NN_ANSI_C=y" in config
    result = dict(status=status, reference_ops=ops, trace_tensors=trace, single_round_fc=single, target="esp32p4", physical_board=False, kernel="esp-nn ANSI C",
         streams=observed, physical_steps=blocks, calls=[normal, expanded], score_errors=errors, event_errors=event_errors,
         differences_source_step_lane_expected_actual=differences,
         reserved_tensor_bytes_per_lane=65536, reserved_variable_bytes_per_lane=16384,
         used_tensor_bytes=[t0,t1], used_variable_allocator_bytes=[v0,v1], simulated_max_us=max_us,
         simulated_blocks_ge_30ms=over, min_internal_heap=heap, uart_sha256=sha(log), reference_sha256=sha(ref),
         fixture_sha256=sha(fixture), config_sha256=sha(WORK / "sdkconfig"), qemu_sha256=sha(QEMU),
         model_sha256=sha(RESULTS / "4090-c161-midlevel-single-v1.tflite"),
         elf_sha256=sha(build / "nyx_midlevel_selftest.elf"), images={name:sha(build / name) for name in files.values()})
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(match[0])
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
