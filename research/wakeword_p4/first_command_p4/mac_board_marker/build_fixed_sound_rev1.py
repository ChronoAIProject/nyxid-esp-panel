"""Build and inspect the 10.1-inch rev1 receiver for the generated Hye Nyx sound."""
import json
from pathlib import Path
import subprocess

from build_rev1_dev import REFERENCE_BARGE
from build_willow_rev1 import MODEL, PATCH, digest, reference_image, run_build

FIXED = PATCH.with_name("fixed-sound-runtime.patch")


def verify(board, build, services):
    desc, app, elf, metadata = reference_image(board, build, services, "fixed-nyx")
    nm = Path(desc["c_compiler"]).with_name("riscv32-esp-elf-nm")
    listing = subprocess.check_output([nm, "--defined-only", elf], text=True).splitlines()
    symbols = {line.split()[-1] for line in listing if line.split()}
    assert {"board_marker_feed_fixed", "board_marker_begin", "board_marker_bytes"} <= symbols
    assert symbols.isdisjoint({"panel_wake_runtime_iface", "panel_wake_runtime_prime_samples",
                               "_binary_hey_nyx_model_bin_start", "_binary_panel_hi_esp_bin_start",
                               "_binary_panel_hey_willow_bin_start", "esp_sr_wakenet9_quantized"})
    model = (board / "models/hey-nyx-c161/model.tflite").read_bytes()
    assert digest(model) == MODEL and model not in app
    assert b"policy=fixed-sound ref=physical tx=unavailable" in app
    for field in ("first_pcm epoch=", "peer_send epoch=", "decision epoch="):
        assert field.encode() in app
    commands = json.loads((build/"compile_commands.json").read_text())
    for name in ("panel_voice.c", "board_marker.cpp", "panel_update.c"):
        command, = [c["command"] for c in commands if Path(c["file"]).name == name]
        assert all("-D"+define+"=1" in command for define in
                   ("NYX_PANEL_FIXED_SOUND","NYX_PANEL_RESEARCH_FIXED_MARKER","NYX_PANEL_UPDATE_DISABLED"))
        assert "-DNYX_PANEL_EXPERIMENTAL_WAKE_RUNTIME" not in command
    return dict(**metadata, engine="fixed-nyx", fixed_sound_patch_sha256=digest(FIXED.read_bytes()),
                reference_model_sha256=MODEL, embedded_c161_copies=0, neural_prime_samples=0)


if __name__ == "__main__":
    run_build("fixed-nyx", FIXED, verify, __doc__, REFERENCE_BARGE)
