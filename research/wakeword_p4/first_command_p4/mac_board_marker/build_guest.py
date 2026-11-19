"""Reuse the existing middle-uart IDF build to link a separate digital P4 fixture."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import check_replay

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("sim_build", type=Path, help="Built by build_panel_marker_order.sh middle-uart")
    parser.add_argument("work", type=Path)
    parser.add_argument("--idf-python", default="python")
    args = parser.parse_args()
    board, sim, work = (p.resolve() for p in (args.board, args.sim_build, args.work))
    service = board / "components/panel_services"
    runtime = board / "components/panel_wake_runtime"
    work.mkdir(exist_ok=True, parents=True)
    check_replay.write_feeders(service, work)
    (work / "guest.c").write_bytes((HERE / "guest.c").read_bytes())
    commands = json.loads((sim / "compile_commands.json").read_text())
    sources = [("panel_voice_audio.c", work / "guest.c"),
               ("listener.cpp", service / "fixed_marker/board_marker.cpp"),
               ("panel_wake_runtime.cpp", runtime / "panel_wake_runtime.cpp")]
    objects = []
    for template, source in sources:
        command, = [shlex.split(c["command"]) for c in commands if Path(c["file"]).name == template]
        target = work / (source.name + ".o")
        subprocess.run([command[0], "-I"+str(work), "-I"+str(runtime), "-I"+str(service),
                        "-DNYX_PANEL_RESEARCH_FIXED_MARKER=1", *command[1:command.index("-o")],
                        "-o", target, "-c", source], cwd=sim, check=True)
        objects.append(str(target))

    # Keep the simulator's IDF link order and test RAM configuration. All new
    # outputs go in work. The original board and simulator ELFs stay available.
    listing = subprocess.check_output(["ninja", "-t", "commands", "nyx_mac_listener.elf"], cwd=sim, text=True)
    command = shlex.split(listing.splitlines()[-1])
    assert command[:2] == [":", "&&"] and command[-2:] == ["&&", ":"]
    link = command[2:-2]
    index = link.index("-o")
    link[index:index+2] = [*objects, "-Wl,--wrap=app_main", "-o", str(work / "board.elf")]
    link = ["-Wl,--Map="+str(work / "board.map") if s.startswith("-Wl,--Map=") else s for s in link]
    subprocess.run(link, cwd=sim, check=True)
    subprocess.run([args.idf_python, "-m", "esptool", "--chip", "esp32p4", "elf2image",
                    "--flash_mode", "dio", "--flash_freq", "80m", "--flash_size", "4MB",
                    "--min-rev-full", "301", "-o", work / "board.bin", work / "board.elf"], check=True)
    model = (board / "models/hey-nyx-c161/model.tflite").read_bytes()
    assert (work / "board.bin").read_bytes().count(model) == 1


if __name__ == "__main__":
    main()
