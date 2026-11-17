#!/usr/bin/env python3
"""Board catalog shared by configuration, packaging, flashing and CI."""
import argparse
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "boards/catalog.json"
IMAGE_BOARD_OFFSET = 32 + 256  # IDF image/segment headers and esp_app_desc_t.
IMAGE_BOARD_SIZE = 112
IMAGE_BOARD_MAGIC = b"NYX-BOARD-V1".ljust(16, b"\0")


def catalog():
    data = json.loads(CATALOG.read_text())
    if data.get("schema") != 1:
        raise ValueError("Unsupported board catalog schema")
    for name, profile in data["profiles"].items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,46}", name):
            raise ValueError("Invalid profile identity")
        board = data["boards"][profile["board"]]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,46}", profile["board"]):
            raise ValueError("Invalid board identity")
        if not re.fullmatch(r"[a-z0-9_]+", board["backend"]):
            raise ValueError("Invalid board backend")
        if not re.fullmatch(r"[a-z0-9_-]+", board["bsp_component"]):
            raise ValueError("Invalid BSP component")
        for filename in [*defaults(name, data), board["bsp_adapter"],
                         f"boards/{board['backend']}/product_backend.c"]:
            if not (ROOT / filename).is_file():
                raise ValueError(f"Missing defaults file: {filename}")
    return data


def defaults(profile, data=None):
    data = data if data is not None else catalog()
    p = data["profiles"][profile]
    return [data["common_defaults"][0], p["revision_defaults"],
            *data["common_defaults"][1:], data["boards"][p["board"]]["defaults"]]


def resolve(profile):
    data = catalog()
    if profile not in data["profiles"]:
        raise ValueError(f"Unknown board profile: {profile}")
    p = data["profiles"][profile]
    return {**p, **data["boards"][p["board"]], "profile": profile}


def validate_config(config, profile):
    p = resolve(profile)
    expected = {"IDF_TARGET": "esp32p4", "ESPTOOLPY_FLASHSIZE": "16MB",
                "ESP32P4_REV_MIN_FULL": p["min_revision"],
                "ESP32P4_REV_MAX_FULL": p["max_revision"],
                "BOOTLOADER_APP_ROLLBACK_ENABLE": True,
                "LV_DRAW_SW_DRAW_UNIT_CNT": 1, **p["config"]}
    for key, value in expected.items():
        actual = config.get(key, False)
        if actual != value:
            raise ValueError(f"{profile}: configuration {key}={actual!r}; expected {value!r}. "
                             "Use a separate build directory/sdkconfig for each profile.")
    return p


def image_identity(image):
    if len(image) < IMAGE_BOARD_OFFSET + IMAGE_BOARD_SIZE or image[0] != 0xE9:
        raise ValueError("Missing board identity in application image")
    desc = image[IMAGE_BOARD_OFFSET:IMAGE_BOARD_OFFSET + IMAGE_BOARD_SIZE]
    if desc[:16] != IMAGE_BOARD_MAGIC:
        raise ValueError("Missing or unsupported board identity in application image")
    fields = []
    for field in (desc[16:64], desc[64:112]):
        text, terminator, padding = field.partition(b"\0")
        if not terminator or any(padding):
            raise ValueError("Malformed image board identity")
        fields.append(text.decode("ascii"))
    p = resolve(fields[0])
    if fields[1] != p["board"]:
        raise ValueError("Image board identity does not match its profile")
    chip = struct.unpack_from("<H", image, 12)[0]
    minimum, maximum = struct.unpack_from("<HH", image, 15)
    if chip != 0x12 or (minimum, maximum) != (p["min_revision"], p["max_revision"]):
        raise ValueError("Image chip/revision header does not match its board profile")
    return p


def validate_image(image, profile):
    p = image_identity(image)
    if p["profile"] != profile:
        raise ValueError(f"Image profile {p['profile']} does not match {profile}")
    return p


def write_header(profile, output):
    p = resolve(profile)
    defines = {"PANEL_BOARD_PROFILE": profile, "PANEL_BOARD_ID": p["board"],
               "PANEL_BOARD_MIN_REVISION": p["min_revision"],
               "PANEL_BOARD_MAX_REVISION": p["max_revision"],
               "PANEL_BOARD_NAME": p["name"], "PANEL_BOARD_BACKEND": p["backend"],
               "PANEL_BOARD_DISPLAY_WIDTH": p["display"]["width"],
               "PANEL_BOARD_DISPLAY_HEIGHT": p["display"]["height"],
               "PANEL_BOARD_AUDIO_INPUT": p["audio"]["input"],
               "PANEL_BOARD_AUDIO_OUTPUT": p["audio"]["output"],
               "PANEL_BOARD_CAPTURE_LAYOUT": p["audio"]["capture"],
               "PANEL_BOARD_CAMERA_SENSOR": p["camera"]["sensor"],
               "PANEL_BOARD_CAMERA_OPTIONAL": int(p["camera"]["optional"]),
               "PANEL_BOARD_CAMERA_WIDTH": p["camera"]["width"],
               "PANEL_BOARD_CAMERA_HEIGHT": p["camera"]["height"],
               "PANEL_BOARD_STYLESHEET_HEADER": f"systems/phone/stylesheets/{p['display']['stylesheet']}/dark/stylesheet.hpp",
               "PANEL_BOARD_STYLESHEET": f"STYLESHEET_{p['display']['stylesheet']}_DARK"}
    lines = ["// Generated from boards/catalog.json. Do not edit.", "#pragma once"]
    for key, value in defines.items():
        encoded = value if key == "PANEL_BOARD_STYLESHEET" else json.dumps(value)
        lines.append(f"#define {key} {encoded}")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    if not output.exists() or output.read_text() != content:
        output.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("matrix", "defaults", "backend", "bsp-component", "bsp-adapter", "bsp-patch", "header", "validate"))
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.action == "matrix":
        print(json.dumps({"profile": list(catalog()["profiles"])}))
        return
    if not args.profile:
        parser.error("--profile is required")
    if args.action == "defaults":
        print(";".join(defaults(args.profile)))
    elif args.action == "backend":
        print(resolve(args.profile)["backend"])
    elif args.action in ("bsp-component", "bsp-adapter", "bsp-patch"):
        print(resolve(args.profile)[args.action.replace("-", "_")])
    elif args.action == "header":
        if not args.output:
            parser.error("--output is required")
        write_header(args.profile, args.output)
    elif args.action == "validate":
        if not args.config:
            parser.error("--config is required")
        validate_config(json.loads(args.config.read_text()), args.profile)


if __name__ == "__main__":
    main()
