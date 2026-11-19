"""Link the wake-initiated mic_task branch from the Mac handoff into the P4 codec fixture."""
import argparse
import json
from pathlib import Path

from build_guest import HERE, link_guest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("codec_build", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--idf-python", default="python")
    args = parser.parse_args()
    service = args.board.resolve() / "components/panel_services"
    sim, work = args.codec_build.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    voice = (service / "panel_voice.c").read_text()
    start = voice.index("static void mic_task(void *arg)\n{")
    stop = voice.index("    // Communication AEC", start)
    (work / "mic_wake.inc").write_text(voice[start:stop] + "    abort();\n}\n")
    constants = [line for line in voice.splitlines() if line.startswith("#define ") and
                 line.split()[1] in ("FRAME_SAMPLES", "WAKE_BUFFER_SAMPLES", "WAKE_LOOKBACK_SAMPLES", "DONE_MIC")]
    assert len(constants) == 4
    (work / "sender_constants.h").write_text("\n".join(constants) + "\n")
    sources = [("panel_voice_audio.c", HERE / "sender.c"), ("panel_voice_audio.c", service / "panel_voice_audio.c")]
    link_guest(sim, work, "nyx_first_command.elf", sources, [work, service], args.idf_python)
    flash = json.loads((sim / "flasher_args.json").read_text())["flash_files"]
    flash = {at: ("board.bin" if name == "nyx_first_command.bin" else str(sim / name)) for at, name in flash.items()}
    (work / "flasher_args.json").write_text(json.dumps(dict(flash_files=flash), indent=2))


if __name__ == "__main__":
    main()
