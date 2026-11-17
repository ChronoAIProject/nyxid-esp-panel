import sys
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
import board_profiles as boards
import build_profile


class BoardProfileTest(unittest.TestCase):
    def config(self, profile):
        p = boards.resolve(profile)
        return {"IDF_TARGET": "esp32p4", "ESPTOOLPY_FLASHSIZE": "16MB",
                "ESP32P4_REV_MIN_FULL": p["min_revision"], "ESP32P4_REV_MAX_FULL": p["max_revision"],
                "LV_DRAW_SW_DRAW_UNIT_CNT": 1, "BOOTLOADER_APP_ROLLBACK_ENABLE": True, **p["config"]}

    def image(self, profile):
        p = boards.resolve(profile)
        image = bytearray(1024)
        image[0] = 0xe9
        struct.pack_into("<H", image, 12, 0x12)
        struct.pack_into("<HH", image, 15, p["min_revision"], p["max_revision"])
        image[288:400] = (boards.IMAGE_BOARD_MAGIC + profile.encode().ljust(48, b"\0") +
                          p["board"].encode().ljust(48, b"\0"))
        return image

    def test_all_profile_pairs_are_checked_before_build_or_update(self):
        profiles = boards.catalog()["profiles"]
        for source in profiles:
            for target in profiles:
                with self.subTest(source=source, target=target):
                    if source == target:
                        boards.validate_config(self.config(source), target)
                        boards.validate_image(self.image(source), target)
                    else:
                        with self.assertRaises(ValueError):
                            boards.validate_config(self.config(source), target)
                        with self.assertRaises(ValueError):
                            boards.validate_image(self.image(source), target)

    def test_stale_board_flag_and_wrong_panel_cannot_silently_fall_back(self):
        for profile in boards.catalog()["profiles"]:
            for key in boards.resolve(profile)["config"]:
                config = self.config(profile)
                config[key] = not config[key]
                with self.subTest(profile=profile, key=key), self.assertRaises(ValueError):
                    boards.validate_config(config, profile)

    def test_missing_truncated_unknown_and_malformed_image_identities_fail(self):
        image = self.image("p4-rev1-16mb")
        for candidate in [b"", image[:399], b"\0" * 1024,
                          image[:288] + b"X" + image[289:],
                          image[:304] + b"unknown".ljust(48, b"\0") + image[352:],
                          image[:352] + b"X" * 48 + image[400:]]:
            with self.assertRaises(ValueError):
                boards.image_identity(candidate)

    def test_build_paths_and_commands_separate_hardware_and_environment(self):
        paths = set()
        for profile in boards.catalog()["profiles"]:
            for environment in ("production", "local"):
                build, cmd = build_profile.command(profile, environment,
                    "http://192.168.1.20:3001" if environment == "local" else None)
                paths.add(build)
                self.assertIn(f"SDKCONFIG={build / 'sdkconfig'}", cmd)
                self.assertIn(f"PANEL_BOARD_PROFILE={profile}", cmd)
        self.assertEqual(len(paths), len(boards.catalog()["profiles"]) * 2)

    def test_cmake_rejects_reusing_another_profiles_directory_before_idf(self):
        # Exercise the real configure entry point. This gate must not depend on
        # an installed SDK or wait until compilation to notice the wrong board.
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            build.mkdir()
            (build / ".panel-board-profile").write_text("p4-rev3-lcd-x8-16mb")
            result = subprocess.run(["cmake", "-S", str(boards.ROOT), "-B", str(build),
                                     "-DPANEL_BOARD_PROFILE=p4-rev1-16mb"],
                                    text=True, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use a separate build directory", " ".join(result.stderr.split()))

    def test_generated_identity_tracks_profile_without_changing_board_wiring(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile in boards.catalog()["profiles"]:
                output = Path(directory) / profile / "panel_board_config.h"
                boards.write_header(profile, output)
                self.assertIn(f'#define PANEL_BOARD_PROFILE "{profile}"', output.read_text())


if __name__ == "__main__":
    unittest.main()
